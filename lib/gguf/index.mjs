// Ouroboros GGUF Parser - Browser/Runtime Agnostic
// Production-ready module for GGUF file streaming analysis
// No external dependencies - pure JavaScript implementation

const GGUFValueType = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
};

const GGUF_MAGIC_NUMBER = new Uint8Array([0x47, 0x47, 0x55, 0x46]);
const GGUF_DEFAULT_ALIGNMENT = 32;
const HTTP_CHUNK_SIZE = 2 * 10 ** 6;
const HTTP_DATA_LEEWAY = 5 * 10 ** 5;
const HTTP_TOTAL_MAX_SIZE = 50 * 10 ** 6;

const MAX_METADATA_ARRAY_LENGTH = 1_000_000;
const MAX_KV_COUNT = 100_000;
const MAX_TENSOR_COUNT = 10_000_000;
const MAX_STRING_LENGTH = 10_000_000;
const MAX_TENSOR_NDIMS = 8;
const MAX_ARRAY_RECURSION_DEPTH = 4;
const MAX_CHUNK_FETCHES_PER_VALUE = 30;

const GGML_PAD = (x, n) => (x + n - 1) & ~(n - 1);

function isGGUFValueType(n) {
  return typeof GGUFValueType[n] === "number";
}

function isVersion(version) {
  return version === 1 || version === 2 || version === 3;
}

class RangeView {
  constructor(uri, params = {}) {
    this.uri = uri;
    this.params = params;
    this.chunk = 0;
    this.buffer = new ArrayBuffer(0, { maxByteLength: HTTP_TOTAL_MAX_SIZE });
    this.dataView = new DataView(this.buffer);
  }

  get view() {
    return this.dataView;
  }

  async fetchChunk() {
    const range = [this.chunk * HTTP_CHUNK_SIZE, (this.chunk + 1) * HTTP_CHUNK_SIZE - 1];
    const fetchFn = this.params.fetch || fetch;
    const headers = {
      ...(this.params.additionalFetchHeaders || {}),
      Range: `bytes=${range[0]}-${range[1]}`,
    };

    const response = await fetchFn(this.uri, { headers });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} while fetching GGUF chunk`);
    }
    const buf = new Uint8Array(await response.arrayBuffer());
    this.appendBuffer(buf);
    this.chunk += 1;
  }

  appendBuffer(buf) {
    if (ArrayBuffer.prototype.resize) {
      this.buffer.resize((this.chunk + 1) * HTTP_CHUNK_SIZE);
      new Uint8Array(this.buffer).set(buf, this.chunk * HTTP_CHUNK_SIZE);
    } else {
      const newBuffer = new ArrayBuffer((this.chunk + 1) * HTTP_CHUNK_SIZE, {
        maxByteLength: HTTP_TOTAL_MAX_SIZE,
      });
      const arrView = new Uint8Array(newBuffer);
      arrView.set(new Uint8Array(this.buffer));
      arrView.set(buf, this.chunk * HTTP_CHUNK_SIZE);
      this.buffer = newBuffer;
      this.dataView = new DataView(this.buffer);
    }
  }

  async fetchChunkIfNeeded(offset) {
    if (this.dataView.byteLength - offset < HTTP_DATA_LEEWAY) {
      await this.fetchChunk();
    }
  }
}

function readVersionedSize(view, byteOffset, version, littleEndian) {
  switch (version) {
    case 1:
      const n = view.getUint32(byteOffset, littleEndian);
      return { value: BigInt(n), length: 4 };
    case 2:
    case 3:
      return { value: view.getBigUint64(byteOffset, littleEndian), length: 8 };
  }
}

function readString(view, offset, version, littleEndian) {
  const length = readVersionedSize(view, offset, version, littleEndian);
  if (length.value > MAX_STRING_LENGTH) {
    throw new Error(
      `String length ${length.value} exceeds maximum allowed (${MAX_STRING_LENGTH})`
    );
  }
  const off = length.length;
  const value = new TextDecoder().decode(
    view.buffer.slice(offset + off, offset + off + Number(length.value))
  );
  return { value, length: off + Number(length.value) };
}

function readMetadataValue(view, type, offset, version, littleEndian, depth = 0) {
  switch (type) {
    case GGUFValueType.UINT8:
      return { value: view.getUint8(offset), length: 1 };
    case GGUFValueType.INT8:
      return { value: view.getInt8(offset), length: 1 };
    case GGUFValueType.UINT16:
      return { value: view.getUint16(offset, littleEndian), length: 2 };
    case GGUFValueType.INT16:
      return { value: view.getInt16(offset, littleEndian), length: 2 };
    case GGUFValueType.UINT32:
      return { value: view.getUint32(offset, littleEndian), length: 4 };
    case GGUFValueType.INT32:
      return { value: view.getInt32(offset, littleEndian), length: 4 };
    case GGUFValueType.FLOAT32:
      return { value: view.getFloat32(offset, littleEndian), length: 4 };
    case GGUFValueType.BOOL:
      return { value: view.getUint8(offset) !== 0, length: 1 };
    case GGUFValueType.STRING:
      return readString(view, offset, version, littleEndian);
    case GGUFValueType.ARRAY: {
      if (depth >= MAX_ARRAY_RECURSION_DEPTH) {
        throw new Error(
          `Nested ARRAY depth ${depth} exceeds maximum allowed (${MAX_ARRAY_RECURSION_DEPTH})`
        );
      }
      const arrayType = view.getUint32(offset, littleEndian);
      if (!isGGUFValueType(arrayType)) {
        throw new Error(`Unsupported array element type: ${arrayType}`);
      }
      const arrayLength = readVersionedSize(view, offset + 4, version, littleEndian);
      if (arrayLength.value > MAX_METADATA_ARRAY_LENGTH) {
        throw new Error(
          `Metadata array length ${arrayLength.value} exceeds maximum allowed (${MAX_METADATA_ARRAY_LENGTH})`
        );
      }
      let length = 4 + arrayLength.length;
      const arrayValues = [];
      for (let i = 0; i < arrayLength.value; i++) {
        const metadataValue = readMetadataValue(view, arrayType, offset + length, version, littleEndian, depth + 1);
        arrayValues.push(metadataValue.value);
        length += metadataValue.length;
      }
      return { value: arrayValues, length };
    }
    case GGUFValueType.UINT64:
      return { value: view.getBigUint64(offset, littleEndian), length: 8 };
    case GGUFValueType.INT64:
      return { value: view.getBigInt64(offset, littleEndian), length: 8 };
    case GGUFValueType.FLOAT64:
      return { value: view.getFloat64(offset, littleEndian), length: 8 };
    default:
      throw new Error(`Unsupported metadata type: ${type}`);
  }
}

function checkBuffer(buffer, header) {
  for (let i = 0; i < header.length; i++) {
    if (header[i] !== buffer[i]) {
      return false;
    }
  }
  return true;
}

export async function gguf(uri, params = {}) {
  const r = new RangeView(uri, params);
  await r.fetchChunk();

  if (!checkBuffer(new Uint8Array(r.view.buffer.slice(0, 4)), GGUF_MAGIC_NUMBER)) {
    throw new Error("not a valid gguf file: not starting with GGUF magic number");
  }

  const [littleEndian, version] = (() => {
    const version = r.view.getUint32(4, true);
    if (version & 65535) {
      return [true, version];
    } else {
      return [false, r.view.getUint32(4, false)];
    }
  })();

  if (!isVersion(version)) {
    throw new Error(`not a valid gguf file: unsupported version "${version}"`);
  }

  let offset = 8;
  const tensorCount = readVersionedSize(r.view, offset, version, littleEndian);
  if (tensorCount.value > MAX_TENSOR_COUNT) {
    throw new Error(`Tensor count ${tensorCount.value} exceeds maximum allowed (${MAX_TENSOR_COUNT})`);
  }
  offset += tensorCount.length;

  const numKv = readVersionedSize(r.view, offset, version, littleEndian);
  if (numKv.value > MAX_KV_COUNT) {
    throw new Error(`KV metadata count ${numKv.value} exceeds maximum allowed (${MAX_KV_COUNT})`);
  }
  offset += numKv.length;

  const metadata = {
    version,
    tensor_count: tensorCount.value,
    kv_count: numKv.value,
  };

  let typedMetadata;
  if (params.typedMetadata) {
    typedMetadata = {
      version: { value: version, type: GGUFValueType.UINT32 },
      tensor_count: {
        value: tensorCount.value,
        type: version === 1 ? GGUFValueType.UINT32 : GGUFValueType.UINT64,
      },
      kv_count: {
        value: numKv.value,
        type: version === 1 ? GGUFValueType.UINT32 : GGUFValueType.UINT64,
      },
    };
  }

  for (let i = 0; i < numKv.value; i++) {
    await r.fetchChunkIfNeeded(offset);

    const keyResult = readString(r.view, offset, version, littleEndian);
    offset += keyResult.length;

    const valueType = r.view.getUint32(offset, littleEndian);
    offset += 4;

    if (!isGGUFValueType(valueType)) {
      throw new Error("Unsupported metadata type: " + valueType);
    }

    let valueResult;
    let fetchCount = 0;
    while (!valueResult) {
      try {
        valueResult = readMetadataValue(r.view, valueType, offset, version, littleEndian);
      } catch (err) {
        if (err instanceof RangeError) {
          if (++fetchCount > MAX_CHUNK_FETCHES_PER_VALUE) {
            throw new Error(
              `Exceeded maximum chunk fetches (${MAX_CHUNK_FETCHES_PER_VALUE}) while reading metadata value`
            );
          }
          await r.fetchChunk();
        } else {
          throw err;
        }
      }
    }
    offset += valueResult.length;
    metadata[keyResult.value] = valueResult.value;

    if (typedMetadata) {
      const typedEntry = {
        value: valueResult.value,
        type: valueType,
      };

      if (valueType === GGUFValueType.ARRAY) {
        const arrayTypeOffset = offset - valueResult.length;
        const arraySubType = r.view.getUint32(arrayTypeOffset, littleEndian);
        if (isGGUFValueType(arraySubType)) {
          typedEntry.subType = arraySubType;
        }
      }

      typedMetadata[keyResult.value] = typedEntry;
    }
  }

  const tensorInfoStartOffset = offset;
  const tensorInfos = [];

  for (let i = 0; i < tensorCount.value; i++) {
    await r.fetchChunkIfNeeded(offset);

    const keyResult = readString(r.view, offset, version, littleEndian);
    offset += keyResult.length;

    const nDims = r.view.getUint32(offset, littleEndian);
    if (nDims > MAX_TENSOR_NDIMS) {
      throw new Error(`Tensor n_dims ${nDims} exceeds maximum allowed (${MAX_TENSOR_NDIMS})`);
    }
    offset += 4;

    const shape = [];
    for (let dim = 0; dim < nDims; dim++) {
      const shapeDim = readVersionedSize(r.view, offset, version, littleEndian);
      shape.push(shapeDim.value);
      offset += shapeDim.length;
    }

    const type = r.view.getUint32(offset, littleEndian);
    offset += 4;
    const tensorOffset = r.view.getBigUint64(offset, littleEndian);
    offset += 8;

    tensorInfos.push({
      name: keyResult.value,
      n_dims: nDims,
      shape,
      dtype: type,
      offset: tensorOffset,
    });
  }

  const rawAlignment = metadata["general.alignment"] ?? GGUF_DEFAULT_ALIGNMENT;
  if (typeof rawAlignment === "bigint" && rawAlignment > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`general.alignment value ${rawAlignment} exceeds safe integer range`);
  }
  const alignment = Number(rawAlignment);
  if (alignment <= 0 || !Number.isInteger(alignment)) {
    throw new Error(`general.alignment must be a positive integer, got ${rawAlignment}`);
  }

  const tensorInfoEndBeforePadOffset = offset;
  const tensorDataOffset = BigInt(GGML_PAD(offset, alignment));

  const baseResult = {
    metadata,
    tensorInfos,
    tensorDataOffset,
    littleEndian,
    tensorInfoByteRange: [tensorInfoStartOffset, tensorInfoEndBeforePadOffset],
  };

  if (params.computeParametersCount && params.typedMetadata) {
    const parameterCount = tensorInfos
      .map(({ shape }) => shape.reduce((acc, val) => acc * Number(val), 1))
      .reduce((acc, val) => acc + val, 0);
    return { ...baseResult, parameterCount, typedMetadata };
  } else if (params.computeParametersCount) {
    const parameterCount = tensorInfos
      .map(({ shape }) => shape.reduce((acc, val) => acc * Number(val), 1))
      .reduce((acc, val) => acc + val, 0);
    return { ...baseResult, parameterCount };
  } else if (params.typedMetadata) {
    return { ...baseResult, typedMetadata };
  } else {
    return baseResult;
  }
}

export { GGUFValueType, GGUF_MAGIC_NUMBER };