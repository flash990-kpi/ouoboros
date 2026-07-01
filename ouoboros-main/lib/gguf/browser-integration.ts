/**
 * Ouroboros Browser Integration Layer
 * Seamless integration between GGUF parser and A.S.T.S. topology system
 * 
 * This module bridges the GGUF parser with Ouroboros kernel for:
 * - Metadata extraction for topology generation
 * - Tensor streaming coordination
 * - Hardware-accelerated inference preparation
 */

import type { GGUFParseOutput, GGUFTensorInfo, GGUFParseOptions } from './index.d';
import type { gguf } from './index.mjs';

/**
 * Browser GGUF Analysis Result
 * Contains everything A.S.T.S. needs for topology synthesis
 */
export interface BrowserGGUFAnalysis {
  modelName: string;
  architecture: string;
  version: number;
  parameterCount: number;
  tensorCount: bigint;
  tensors: Array<{
    name: string;
    shape: bigint[];
    sizeBytes: bigint;
    quantizationType: string;
  }>;
  metadata: Record<string, unknown>;
  tensorDataOffset: bigint;
  littleEndian: boolean;
  tensorInfoByteRange: [number, number];
}

/**
 * Parse GGUF in browser and extract A.S.T.S.-compatible analysis
 * Non-blocking, streaming, memory-efficient
 */
export async function analyzeBrowserGGUF(
  ggufUri: string,
  options?: {
    fetch?: typeof fetch;
    headers?: Record<string, string>;
    progressCallback?: (progress: { loaded: number; total: number }) => void;
  }
): Promise<BrowserGGUFAnalysis> {
  const { gguf } = await import('./index.mjs');

  try {
    const parseOptions: GGUFParseOptions = {
      fetch: options?.fetch,
      additionalFetchHeaders: options?.headers,
      computeParametersCount: true,
      typedMetadata: true,
    };

    const parseResult = await gguf(ggufUri, parseOptions);

    const modelName =
      (parseResult.metadata["general.name"] as string) || "Unknown-Model";
    const architecture =
      (parseResult.metadata["general.architecture"] as string) || "unknown";

    // Calculate total tensor size
    const totalTensorSize = parseResult.tensorInfos.reduce((sum, tensor) => {
      const size = tensor.shape.reduce((acc, dim) => acc * Number(dim), 1n);
      return sum + size;
    }, 0n);

    return {
      modelName,
      architecture,
      version: parseResult.metadata.version as number,
      parameterCount: parseResult.parameterCount || 0,
      tensorCount: parseResult.metadata.tensor_count,
      tensors: parseResult.tensorInfos.map((tensor: GGUFTensorInfo) => ({
        name: tensor.name,
        shape: tensor.shape,
        sizeBytes: tensor.shape.reduce((acc, dim) => acc * Number(dim), 1n),
        quantizationType: getQuantizationName(tensor.dtype),
      })),
      metadata: parseResult.metadata,
      tensorDataOffset: parseResult.tensorDataOffset,
      littleEndian: parseResult.littleEndian,
      tensorInfoByteRange: parseResult.tensorInfoByteRange,
    };
  } catch (error) {
    throw new Error(
      `Failed to analyze GGUF: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Map GGML quantization type ID to human-readable name
 */
function getQuantizationName(dtype: number): string {
  const quantTypes: Record<number, string> = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "Q8_K",
  };
  return quantTypes[dtype] || `UNKNOWN(${dtype})`;
}

/**
 * Stream GGUF tensor chunk for A.S.T.S. topology synthesis
 * Used by weightSynthesizer to fetch specific weight clusters
 */
export async function streamTensorCluster(
  ggufUri: string,
  tensorName: string,
  clusterIndex: number,
  clusterSize: number,
  options?: {
    fetch?: typeof fetch;
    headers?: Record<string, string>;
  }
): Promise<ArrayBuffer> {
  try {
    const response = await (options?.fetch || fetch)(ggufUri, {
      headers: {
        ...(options?.headers || {}),
        Range: `bytes=${clusterIndex * clusterSize}-${(clusterIndex + 1) * clusterSize - 1}`,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: Failed to fetch tensor cluster`);
    }

    return await response.arrayBuffer();
  } catch (error) {
    throw new Error(
      `Failed to stream tensor cluster: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Validate GGUF file integrity
 * Checks magic number and version compatibility
 */
export async function validateGGUFFile(
  ggufUri: string,
  options?: {
    fetch?: typeof fetch;
  }
): Promise<{ valid: boolean; reason?: string; version?: number }> {
  try {
    const response = await (options?.fetch || fetch)(ggufUri, {
      headers: { Range: "bytes=0-15" },
    });

    if (!response.ok) {
      return { valid: false, reason: `HTTP ${response.status}` };
    }

    const buffer = await response.arrayBuffer();
    const view = new Uint8Array(buffer);

    // Check magic number
    if (
      view[0] !== 0x47 ||
      view[1] !== 0x47 ||
      view[2] !== 0x55 ||
      view[3] !== 0x46
    ) {
      return { valid: false, reason: "Invalid GGUF magic number" };
    }

    // Check version
    const versionView = new DataView(buffer);
    const version = versionView.getUint32(4, true);
    const isValidVersion = version === 1 || version === 2 || version === 3;

    if (!isValidVersion) {
      return { valid: false, reason: `Unsupported GGUF version ${version}` };
    }

    return { valid: true, version };
  } catch (error) {
    return {
      valid: false,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}