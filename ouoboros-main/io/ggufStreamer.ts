import { gguf } from '@huggingface/gguf';
import { TopologyParser } from '../asts/topologyParser.js';

export interface StreamSource {
    type: 'LOCAL' | 'REMOTE';
    fileObject?: File;
    remoteUrl?: string;
}

export class GgufStreamer {
    private source: StreamSource;

    constructor(source: StreamSource) {
        if (source.type === 'LOCAL' && !source.fileObject) {
            throw new Error("L'origine di streaming locale richiede l'assegnazione dell'oggetto File del browser.");
        }
        if (source.type === 'REMOTE' && !source.remoteUrl) {
            throw new Error("L'origine di streaming remota richiede la specifica di un URL endpoint valido.");
        }
        this.source = source;
    }

    // Metodo per leggere chunk di peso (usato dall'inferenza)
    public async readWeightChunk(offset: bigint, length: bigint): Promise<ArrayBuffer> {
        const numOffset = Number(offset);
        const numLength = Number(length);

        if (this.source.type === 'LOCAL') {
            return await this.readLocalSlice(numOffset, numLength);
        } else {
            return await this.readRemoteRange(numOffset, numLength);
        }
    }

    // Genera l'indice topologico .ouro dal file GGUF
    public async generateTopologyFromGguf(): Promise<ArrayBuffer> {
        const file = this.source.fileObject!;
        const blobUrl = URL.createObjectURL(file);

        try {
            // Usa la libreria @huggingface/gguf per parsare il file
            const { tensorInfos } = await gguf(blobUrl);

            const tensorCount = tensorInfos.length;
            let maxLayerFound = 0;

            // Buffer .ouro: magic + version + tensorCount + layerCount + record per tensore (48 byte ciascuno)
            const ouroBuffer = new ArrayBuffer(32 + (tensorCount * 48));
            const ouroView = new DataView(ouroBuffer);

            // Magic "OURO"
            ouroView.setUint32(0, 0x4f55524f, true);
            // Version
            ouroView.setUint32(4, 1, true);
            // Tensor count
            ouroView.setUint32(8, tensorCount, true);

            let ouroCursor = 32;

            for (const tensor of tensorInfos) {
                const name = tensor.name;
                const offset = BigInt(tensor.offset || 0);
                const shape = tensor.shape || [];
                const dtype = tensor.dtype;

                // Calcola lunghezza in byte
                let totalElements = 1;
                for (const dim of shape) {
                    totalElements *= Number(dim);
                }

                // Mappa dtype a dimensione in byte (valori numerici)
                const dtypeMap: Record<number, number> = {
                    0: 4,  // F32
                    1: 2,  // F16
                    2: 2,  // Q4_0
                    3: 2,  // Q4_1
                    4: 2,  // Q5_0
                    5: 2,  // Q5_1
                    6: 1,  // Q8_0
                    7: 1,  // Q8_1
                    8: 4,  // Q2_K
                    9: 4,  // Q3_K
                    10: 4, // Q4_K
                    11: 4, // Q5_K
                    12: 4, // Q6_K
                    13: 4, // Q8_K
                    14: 1, // I8
                    15: 2, // I16
                    16: 4, // I32
                    17: 8, // I64
                };
                const dtypeNumber = typeof dtype === 'number' ? dtype : 0;
                const elementSize = dtypeMap[dtypeNumber] || 4;
                const byteLength = BigInt(totalElements * elementSize);

                // Estrai layer index dal nome (es. "blk.0.attn_q.weight")
                const layerMatch = name.match(/blk\.(\d+)\./);
                const layerIndex = layerMatch ? parseInt(layerMatch[1], 10) : 0;
                if (layerIndex > maxLayerFound) maxLayerFound = layerIndex;

                // Sparsity rank (euristica)
                let sparsityRank = 4;
                if (name.includes("ffn_down") || name.includes("v_proj")) sparsityRank = 1;
                else if (name.includes("attn_q") || name.includes("attn_k")) sparsityRank = 2;
                else if (name.includes("ffn_up") || name.includes("ffn_gate")) sparsityRank = 3;

                const tensorHash = TopologyParser.hashTensorName(name);

                // Scrittura record
                ouroView.setBigUint64(ouroCursor, tensorHash, true);
                ouroView.setBigUint64(ouroCursor + 8, offset, true);
                ouroView.setBigUint64(ouroCursor + 16, byteLength, true);
                ouroView.setUint32(ouroCursor + 24, layerIndex, true);
                ouroView.setUint32(ouroCursor + 28, dtypeNumber, true);
                ouroView.setUint32(ouroCursor + 32, sparsityRank, true);
                ouroView.setBigUint64(ouroCursor + 36, 0n, true); // riservato
                ouroCursor += 48;
            }

            // Scrittura layer count
            ouroView.setUint32(12, maxLayerFound + 1, true);

            // Salva automaticamente il file .ouro
            this.saveOuroToDisk(ouroBuffer, file.name);

            return ouroBuffer;
        } finally {
            URL.revokeObjectURL(blobUrl);
        }
    }

    private saveOuroToDisk(ouroBuffer: ArrayBuffer, ggufFileName: string): void {
        try {
            const ouroName = ggufFileName.replace(/\.gguf$/i, '.ouro');
            const blob = new Blob([ouroBuffer], { type: 'application/octet-stream' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = ouroName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(() => URL.revokeObjectURL(link.href), 5000);
            console.log(`[OUROBOROS] .ouro salvato: ${ouroName}`);
        } catch (e) {
            console.warn('[OUROBOROS] Salvataggio .ouro fallito:', e);
        }
    }

    private readLocalSlice(offset: number, length: number): Promise<ArrayBuffer> {
        return new Promise((resolve, reject) => {
            const file = this.source.fileObject!;
            const slice = file.slice(offset, offset + length);
            const reader = new FileReader();
            reader.onload = () => {
                if (reader.result instanceof ArrayBuffer) resolve(reader.result);
                else reject(new Error("Errore lettura chunk."));
            };
            reader.onerror = () => reject(reader.error);
            reader.readAsArrayBuffer(slice);
        });
    }

    private async readRemoteRange(offset: number, length: number): Promise<ArrayBuffer> {
        const end = offset + length - 1;
        const response = await fetch(this.source.remoteUrl!, {
            method: 'GET',
            headers: { 'Range': `bytes=${offset}-${end}` }
        });
        if (response.status !== 206 && !response.ok) {
            throw new Error(`HTTP Range non supportato: ${response.status}`);
        }
        return await response.arrayBuffer();
    }
}