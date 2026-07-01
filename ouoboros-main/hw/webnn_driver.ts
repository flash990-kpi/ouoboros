export class WebNnDriver {
    private context: any = null;
    private builder: any = null;

    public async initialize(): Promise<void> {
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
    }

    public async executePayload(weightBuffer: SharedArrayBuffer, inputData: Float32Array): Promise<Float32Array> {
        if (!this.context || !this.builder) {
            throw new Error("Esecuzione bloccata: Sottosistema NPU WebNN non pronto.");
        }

        const inputLen = inputData.length;
        const weightsF32 = new Float32Array(weightBuffer);
        const outputLen = weightsF32.length / inputLen;

        const weightDescriptor = { type: 'float32', dimensions: [outputLen, inputLen] };
        const inputDescriptor = { type: 'float32', dimensions: [1, inputLen] };

        const constantWeights = this.builder.constant(weightDescriptor, weightsF32);
        const nodeInput = this.builder.input('inputTensor', inputDescriptor);
        
        // Moltiplicazione di matrice accelerata direttamente su hardware NPU
        const outputGraphNode = this.builder.matmul(nodeInput, constantWeights);
        const compiledGraph = await this.builder.build({ 'outputTensor': outputGraphNode });

        const finalOutputBuffer = new Float32Array(outputLen);
        await this.context.compute(compiledGraph, { 'inputTensor': inputData }, { 'outputTensor': finalOutputBuffer });

        return finalOutputBuffer;
    }
}
