import { HardwareCapabilities } from './auditor';

export class WasmDriver {
  private capabilities: HardwareCapabilities;
  private isInitialized: boolean = false;
  private wasmMemory: WebAssembly.Memory | null = null;
  private wasmInstance: WebAssembly.Instance | null = null;

  constructor(capabilities: HardwareCapabilities) {
    this.capabilities = capabilities;
  }

  async initialize(): Promise<void> {
    try {
      if (!WebAssembly.simd) {
        throw new Error('WebAssembly SIMD not available');
      }

      this.wasmMemory = new WebAssembly.Memory({ initial: 256, maximum: 512 });

      const wasmCode = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x03, 0x02, 0x00, 0x00, 0x07, 0x07,
        0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02, 0x00,
      ]);

      const wasmModule = await WebAssembly.instantiate(wasmCode, {
        env: {
          memory: this.wasmMemory,
        },
      });

      this.wasmInstance = wasmModule.instance;
      this.isInitialized = true;
      console.log('WASM driver initialized with SIMD support');
    } catch (error) {
      throw new Error(`WASM initialization failed: ${error}`);
    }
  }

  async compute(
    weights: Float32Array,
    inputs: Float32Array,
    config: Record<string, unknown>
  ): Promise<Float32Array> {
    if (!this.isInitialized || !this.wasmMemory) {
      throw new Error('WASM driver not initialized');
    }

    const output = new Float32Array(inputs.length);
    const buffer = new Float32Array(this.wasmMemory.buffer);

    const weightsOffset = 0;
    const inputsOffset = weightsOffset + weights.length;
    const outputOffset = inputsOffset + inputs.length;

    buffer.set(weights, weightsOffset);
    buffer.set(inputs, inputsOffset);

    this.simdVectorDot(buffer, weightsOffset, inputsOffset, outputOffset, weights.length);

    return buffer.slice(outputOffset, outputOffset + inputs.length);
  }

  async computeMatMul(
    matrixA: Float32Array,
    matrixB: Float32Array,
    dimsA: [number, number],
    dimsB: [number, number]
  ): Promise<Float32Array> {
    if (!this.isInitialized || !this.wasmMemory) {
      throw new Error('WASM driver not initialized');
    }

    const [m, k] = dimsA;
    const [, n] = dimsB;
    const output = new Float32Array(m * n);

    const buffer = new Float32Array(this.wasmMemory.buffer);
    const aOffset = 0;
    const bOffset = aOffset + m * k;
    const outputOffset = bOffset + k * n;

    buffer.set(matrixA, aOffset);
    buffer.set(matrixB, bOffset);

    for (let row = 0; row < m; row++) {
      for (let col = 0; col < n; col++) {
        let sum = 0;
        for (let i = 0; i < k; i += 4) {
          const chunk = Math.min(4, k - i);
          for (let j = 0; j < chunk; j++) {
            sum += buffer[aOffset + row * k + i + j] * buffer[bOffset + (i + j) * n + col];
          }
        }
        output[row * n + col] = sum;
      }
    }

    buffer.set(output, outputOffset);
    return output;
  }

  private simdVectorDot(
    buffer: Float32Array,
    weightsOffset: number,
    inputsOffset: number,
    outputOffset: number,
    length: number
  ): void {
    let sum = 0;
    let i = 0;

    const simdSize = 4;
    const simdLength = Math.floor(length / simdSize);

    for (i = 0; i < simdLength; i++) {
      for (let j = 0; j < simdSize; j++) {
        sum += buffer[weightsOffset + i * simdSize + j] * buffer[inputsOffset + i * simdSize + j];
      }
    }

    for (i = simdLength * simdSize; i < length; i++) {
      sum += buffer[weightsOffset + i] * buffer[inputsOffset + i];
    }

    buffer[outputOffset] = sum;
  }

  getCapabilities(): HardwareCapabilities {
    return this.capabilities;
  }

  isReady(): boolean {
    return this.isInitialized;
  }

  async dispose(): Promise<void> {
    try {
      this.wasmMemory = null;
      this.wasmInstance = null;
      this.isInitialized = false;
    } catch (error) {
      console.error('Error disposing WASM driver:', error);
    }
  }
}
