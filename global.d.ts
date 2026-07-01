// Dichiarazioni minime per WebGPU (usate nel codice)
declare const GPUBufferUsage: {
    STORAGE: number;
    COPY_SRC: number;
    COPY_DST: number;
    MAP_READ: number;
    // Aggiungi altri se necessario
};

declare const GPUMapMode: {
    READ: number;
};

// Aggiungi interfacce per evitare errori su altri tipi usati
interface Navigator {
    gpu?: {
        requestAdapter(options?: any): Promise<any>;
    };
}

interface GPUDevice {
    createBuffer(descriptor: any): any;
    createComputePipeline(descriptor: any): any;
    createShaderModule(descriptor: any): any;
    createBindGroup(descriptor: any): any;
    createCommandEncoder(): any;
    queue: any;
}

interface GPUBuffer {
    getMappedRange(): ArrayBuffer;
    unmap(): void;
    mapAsync(mode: number): Promise<void>;
    destroy(): void;
}

// Aggiungi altre interfacce se necessario per evitare errori su GPUBufferUsage ecc.
