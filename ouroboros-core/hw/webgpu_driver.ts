import { HardwareCapabilities } from './auditor';

export class WebGPUDriver {
  private device: GPUDevice | null = null;
  private queue: GPUQueue | null = null;
  private capabilities: HardwareCapabilities;
  private isInitialized: boolean = false;
  private shaderCache: Map<string, GPUShaderModule> = new Map();

  constructor(capabilities: HardwareCapabilities) {
    this.capabilities = capabilities;
  }

  async initialize(): Promise<void> {
    try {
      if (!navigator.gpu) {
        throw new Error('WebGPU not available');
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No GPU adapter found');

      this.device = await adapter.requestDevice();
      this.queue = this.device.queue;
      this.isInitialized = true;
      console.log('WebGPU driver initialized');
    } catch (error) {
      throw new Error(`WebGPU initialization failed: ${error}`);
    }
  }

  async compute(
    weights: Float32Array,
    inputs: Float32Array,
    config: Record<string, unknown>
  ): Promise<Float32Array> {
    if (!this.isInitialized || !this.device) {
      throw new Error('WebGPU driver not initialized');
    }

    const weightsBuffer = this.device.createBuffer({
      size: weights.byteLength,
      mappedAtCreation: true,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    new Float32Array(weightsBuffer.getMappedRange()).set(weights);
    weightsBuffer.unmap();

    const inputsBuffer = this.device.createBuffer({
      size: inputs.byteLength,
      mappedAtCreation: true,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    new Float32Array(inputsBuffer.getMappedRange()).set(inputs);
    inputsBuffer.unmap();

    const outputBuffer = this.device.createBuffer({
      size: inputs.byteLength,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    });

    const shaderCode = `
      @group(0) @binding(0) var<storage, read_write> inputs: array<f32>;
      @group(0) @binding(1) var<storage, read_write> weights: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx >= arrayLength(&output)) { return; }
        var sum = 0.0;
        for (var i = 0u; i < arrayLength(&weights); i = i + 1u) {
          sum += inputs[i] * weights[i];
        }
        output[idx] = sum;
      }
    `;

    const shaderModule = this.device.createShaderModule({
      code: shaderCode,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: inputsBuffer } },
        { binding: 1, resource: { buffer: weightsBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'main' },
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(inputs.length / 256));
    passEncoder.end();

    const stagingBuffer = this.device.createBuffer({
      size: inputs.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, inputs.byteLength);
    this.queue!.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange()).slice();
    stagingBuffer.unmap();

    weightsBuffer.destroy();
    inputsBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();

    return result;
  }

  async computeMatMul(
    matrixA: Float32Array,
    matrixB: Float32Array,
    dimsA: [number, number],
    dimsB: [number, number]
  ): Promise<Float32Array> {
    if (!this.isInitialized || !this.device) {
      throw new Error('WebGPU driver not initialized');
    }

    const [m, k] = dimsA;
    const [, n] = dimsB;

    const aBuffer = this.device.createBuffer({
      size: matrixA.byteLength,
      mappedAtCreation: true,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    new Float32Array(aBuffer.getMappedRange()).set(matrixA);
    aBuffer.unmap();

    const bBuffer = this.device.createBuffer({
      size: matrixB.byteLength,
      mappedAtCreation: true,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    new Float32Array(bBuffer.getMappedRange()).set(matrixB);
    bBuffer.unmap();

    const outputBuffer = this.device.createBuffer({
      size: m * n * 4,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    });

    const shaderCode = `
      @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
      @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(16, 16)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let row = global_id.x;
        let col = global_id.y;
        let m = 32u;
        let k = 64u;
        let n = 32u;

        if (row >= m || col >= n) { return; }

        var sum = 0.0;
        for (var i = 0u; i < k; i = i + 1u) {
          sum += matrixA[row * k + i] * matrixB[i * n + col];
        }
        output[row * n + col] = sum;
      }
    `;

    const shaderModule = this.device.createShaderModule({
      code: shaderCode,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'main' },
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(m / 16), Math.ceil(n / 16));
    passEncoder.end();

    const stagingBuffer = this.device.createBuffer({
      size: m * n * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, m * n * 4);
    this.queue!.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange()).slice();
    stagingBuffer.unmap();

    aBuffer.destroy();
    bBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();

    return result;
  }

  getCapabilities(): HardwareCapabilities {
    return this.capabilities;
  }

  isReady(): boolean {
    return this.isInitialized;
  }

  async dispose(): Promise<void> {
    try {
      if (this.device) {
        this.shaderCache.clear();
        this.device.destroy();
        this.device = null;
        this.isInitialized = false;
      }
    } catch (error) {
      console.error('Error disposing WebGPU driver:', error);
    }
  }
}
