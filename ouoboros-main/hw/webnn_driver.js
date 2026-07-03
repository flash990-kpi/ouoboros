export class WebNnDriver {
    constructor() {
        this.context = null;
        this.builder = null;
    }
    async initialize(hardwareProfile) {
        console.log('[WEBNN DRIVER] Initializing NPU driver...');
        // @ts-ignore
        if (!navigator.ml) {
            throw new Error("L'interfaccia nativa W3C WebNN non è disponibile in questo browser.");
        }
        // @ts-ignore
        this.context = await navigator.ml.createContext({ deviceType: 'npu' });
        if (!this.context) {
            throw new Error("Inizializzazione del contesto NPU fallita.");
        }
        // @ts-ignore
        this.builder = new MLGraphBuilder(this.context);
        console.log('[WEBNN DRIVER] NPU driver initialized successfully');
    }
    async execute(weights, tensorInfo, routingPath) {
        console.log('[WEBNN DRIVER] Executing on NPU...');
        if (!this.context || !this.builder) {
            throw new Error("Esecuzione bloccata: Sottosistema NPU WebNN non pronto.");
        }
        
        // Converti pesi in Float32Array per NPU
        const weightsF32 = new Float32Array(weights);
        
        // Esegui operazione matriciale su NPU
        const inputLen = tensorInfo.shape.reduce((a, b) => a * b, 1);
        const outputLen = weightsF32.length / inputLen;
        
        const weightDescriptor = { type: 'float32', dimensions: [outputLen, inputLen] };
        const inputDescriptor = { type: 'float32', dimensions: [1, inputLen] };
        
        const constantWeights = this.builder.constant(weightDescriptor, weightsF32);
        const nodeInput = this.builder.input('inputTensor', inputDescriptor);
        
        // Moltiplicazione di matrice accelerata direttamente su hardware NPU
        const outputGraphNode = this.builder.matmul(nodeInput, constantWeights);
        const compiledGraph = await this.builder.build({ 'outputTensor': outputGraphNode });
        
        const finalOutputBuffer = new Float32Array(outputLen);
        await this.context.compute(compiledGraph, { 'inputTensor': new Float32Array([1]) }, { 'outputTensor': finalOutputBuffer });
        
        // Restituisci logits per sampling (non più lettere placeholder)
        return { logits: finalOutputBuffer, output: finalOutputBuffer };
    }
}
