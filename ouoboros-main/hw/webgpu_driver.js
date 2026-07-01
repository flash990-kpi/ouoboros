// ============================================================
// Dichiarazioni delle costanti WebGPU (valori ufficiali)
// ============================================================
const GPUBufferUsage = {
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    INDEX: 0x0010,
    VERTEX: 0x0020,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
    INDIRECT: 0x0100,
    QUERY_RESOLVE: 0x0200,
};
const GPUMapMode = {
    READ: 0x0001,
    WRITE: 0x0002,
};
// ============================================================
// Driver WebGPU
// ============================================================
export class WebGpuDriver {
    constructor() {
        // @ts-ignore - WebGPU types not available in Node.js environment
        this.device = null;
        // @ts-ignore - WebGPU types not available in Node.js environment
        this.pipeline = null;
    }
    async initialize() {
        // @ts-ignore - WebGPU not available in Node.js environment
        if (!navigator.gpu) {
            throw new Error("WebGPU non implementato nel contesto browser corrente.");
        }
        // @ts-ignore - WebGPU not available in Node.js environment
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) {
            throw new Error("Nessun hardware grafico compatibile rilevato dal sottosistema WebGPU.");
        }
        this.device = await adapter.requestDevice();
        this.compileShaderCore();
    }
    compileShaderCore() {
        if (!this.device)
            return;
        const wgslKernel = `
            @group(0) @binding(0) var<storage, read> weights : array<f32>;
            @group(0) @binding(1) var<storage, read> inputVector : array<f32>;
            @group(0) @binding(2) var<storage, read_write> outputVector : array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let currentThread = global_id.x;
                let sizeInput = arrayLength(&inputVector);
                let sizeOutput = arrayLength(&outputVector);

                if (currentThread >= sizeOutput) {
                    return;
                }

                var accumulator: f32 = 0.0;
                let rowOffset = currentThread * sizeInput;

                for (var i: u32 = 0u; i < sizeInput; i = i + 1u) {
                    accumulator = accumulator + (weights[rowOffset + i] * inputVector[i]);
                }

                outputVector[currentThread] = accumulator;
            }
        `;
        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: wgslKernel }),
                entryPoint: 'main'
            }
        });
    }
    async executePayload(weightBuffer, inputData) {
        if (!this.device || !this.pipeline) {
            throw new Error("Impossibile eseguire il payload: Driver WebGPU non inizializzato.");
        }
        const floatArrayWeights = new Float32Array(weightBuffer);
        const outputLength = floatArrayWeights.length / inputData.length;
        if (outputLength <= 0 || outputLength % 1 !== 0) {
            throw new Error(`Incongruenza dimensionale della matrice dei pesi. Elementi: ${floatArrayWeights.length}, Input: ${inputData.length}`);
        }
        const gWeights = this.device.createBuffer({
            size: floatArrayWeights.byteLength,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true
        });
        new Float32Array(gWeights.getMappedRange()).set(floatArrayWeights);
        gWeights.unmap();
        const gInput = this.device.createBuffer({
            size: inputData.byteLength,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true
        });
        new Float32Array(gInput.getMappedRange()).set(inputData);
        gInput.unmap();
        const gOutput = this.device.createBuffer({
            size: outputLength * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const gRead = this.device.createBuffer({
            size: outputLength * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: gWeights } },
                { binding: 1, resource: { buffer: gInput } },
                { binding: 2, resource: { buffer: gOutput } }
            ]
        });
        const cmdEncoder = this.device.createCommandEncoder();
        const passEncoder = cmdEncoder.beginComputePass();
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(outputLength / 256));
        passEncoder.end();
        cmdEncoder.copyBufferToBuffer(gOutput, 0, gRead, 0, outputLength * 4);
        this.device.queue.submit([cmdEncoder.finish()]);
        await gRead.mapAsync(GPUMapMode.READ);
        const outputBufferMapped = new Float32Array(gRead.getMappedRange().slice(0));
        gRead.unmap();
        gWeights.destroy();
        gInput.destroy();
        gOutput.destroy();
        gRead.destroy();
        return outputBufferMapped;
    }
}
