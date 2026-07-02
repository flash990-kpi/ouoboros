import { TopologyParser } from '../asts/topologyParser.js';

// GGUF Type Constants
const GGUF_TYPE = {
    UINT8: 0,
    UINT8: 1,
    UINT16: 2,
    UINT32: 3,
    UINT64: 4,
    INT8: 5,
    INT8: 6,
    INT16: 7,
    INT32: 8,
    INT64: 9,
    FLOAT32: 10,
    FLOAT64: 11,
    BOOL: 12,
    STRING: 13,
    ARRAY: 14,
};

export class GgufStreamer {
    constructor(source) {
        if (source.type === 'LOCAL' && !source.fileObject) {
            throw new Error("L'origine di streaming locale richiede l'assegnazione dell'oggetto File del browser.");
        }
        if (source.type === 'REMOTE' && !source.remoteUrl) {
            throw new Error("L'origine di streaming remota richiede la specifica di un URL endpoint valido.");
        }
        this.source = source;
    }
    
    // Metodo per leggere chunk di peso (usato dall'inferenza)
    async readWeightChunk(offset, length) {
        const numOffset = Number(offset);
        const numLength = Number(length);
        if (this.source.type === 'LOCAL') {
            return await this.readLocalSlice(numOffset, numLength);
        }
        else {
            return await this.readRemoteRange(numOffset, numLength);
        }
    }
    
    // Genera l'indice topologico .ouro dal file GGUF usando @huggingface/gguf (official Hugging Face parser from esm.sh CDN)
    async generateTopologyFromGguf() {
        const file = this.source.fileObject;
        
        try {
            console.log('[GGUF] Using @huggingface/gguf from esm.sh CDN (official Hugging Face parser)...');
            console.log(`[GGUF] File size: ${file.size} bytes`);
            
            // Usa @huggingface/gguf da esm.sh CDN (bundled con tutte le dipendenze)
            const { gguf } = await import('https://esm.sh/@huggingface/gguf@0.4.2');
            
            // Crea URL blob per il file locale (workaround per browser)
            const fileUrl = URL.createObjectURL(file);
            console.log('[GGUF] Parsing GGUF with blob URL...');
            
            const { metadata, tensorInfos } = await gguf(fileUrl);
            
            console.log('[GGUF] Parsed with @huggingface/gguf:', {
                tensorCount: tensorInfos.length,
                metadataKeys: Object.keys(metadata)
            });
            
            // Converti i tensori dal formato @huggingface/gguf al nostro formato
            const tensors = [];
            for (const tensorInfo of tensorInfos) {
                tensors.push({
                    name: tensorInfo.name,
                    shape: tensorInfo.shape.map(dim => BigInt(dim)),
                    dtype: tensorInfo.dtype,
                    offset: BigInt(tensorInfo.offset || 0)
                });
            }
            
            const tensorCount = tensors.length;
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
            
            for (const tensor of tensors) {
                const name = tensor.name;
                const offset = tensor.offset;
                const shape = tensor.shape;
                const dtype = tensor.dtype;
                
                // Calcola lunghezza in byte
                let totalElements = 1;
                for (const dim of shape) {
                    totalElements *= Number(dim);
                }
                
                // Mappa dtype a dimensione in byte
                const dtypeMap = {
                    0: 4, // F32
                    1: 2, // F16
                    2: 2, // Q4_0
                    3: 2, // Q4_1
                    4: 2, // Q5_0
                    5: 2, // Q5_1
                    6: 1, // Q8_0
                    7: 1, // Q8_1
                    8: 4, // Q2_K
                    9: 4, // Q3_K
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
                
                // Estrai layer index dal nome (supporta vari formati)
                const layerMatch = name.match(/blk\.(\d+)\./) || 
                                  name.match(/layers\.(\d+)\./) ||
                                  name.match(/layer\.(\d+)\./);
                const layerIndex = layerMatch ? parseInt(layerMatch[1], 10) : 0;
                if (layerIndex > maxLayerFound)
                    maxLayerFound = layerIndex;
                
                // Sparsity rank (euristica migliorata)
                let sparsityRank = 4;
                if (name.includes("ffn_down") || name.includes("v_proj") || name.includes("down_proj"))
                    sparsityRank = 1;
                else if (name.includes("attn_q") || name.includes("attn_k") || name.includes("q_proj") || name.includes("k_proj"))
                    sparsityRank = 2;
                else if (name.includes("ffn_up") || name.includes("ffn_gate") || name.includes("up_proj") || name.includes("gate_proj"))
                    sparsityRank = 3;
                
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
            
            // Pulisci URL blob
            URL.revokeObjectURL(fileUrl);
            
            // Salva automaticamente il file .ouro
            this.saveOuroToDisk(ouroBuffer, file.name);
            return ouroBuffer;
        }
        catch (error) {
            console.error('[GGUF] Error parsing file:', error);
            throw new Error(
                `Errore nel parsing del file GGUF: ${error.message}. ` +
                'Assicurati che il file sia un GGUF valido.'
            );
        }
    }
    
    saveOuroToDisk(ouroBuffer, ggufFileName) {
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
        }
        catch (e) {
            console.warn('[OUROBOROS] Salvataggio .ouro fallito:', e);
        }
    }
    readLocalSlice(offset, length) {
        return new Promise((resolve, reject) => {
            const file = this.source.fileObject;
            console.log(`[GGUF] Reading slice: offset=${offset}, length=${length}, fileSize=${file.size}`);
            
            // Per file molto grandi, usa chunk più piccoli
            const maxChunkSize = 100 * 1024 * 1024; // 100MB max per chunk
            const actualLength = Math.min(length, maxChunkSize);
            
            const slice = file.slice(offset, offset + actualLength);
            const reader = new FileReader();
            
            reader.onload = () => {
                if (reader.result instanceof ArrayBuffer) {
                    console.log(`[GGUF] Successfully read ${reader.result.byteLength} bytes`);
                    resolve(reader.result);
                } else {
                    reject(new Error("Errore lettura chunk: result non è ArrayBuffer"));
                }
            };
            
            reader.onerror = () => {
                console.error('[GGUF] FileReader error:', reader.error);
                reject(reader.error);
            };
            
            reader.onabort = () => {
                console.error('[GGUF] FileReader aborted');
                reject(new Error("FileReader aborted"));
            };
            
            try {
                reader.readAsArrayBuffer(slice);
            } catch (e) {
                console.error('[GGUF] Exception in readAsArrayBuffer:', e);
                reject(e);
            }
        });
    }
    async readRemoteRange(offset, length) {
        const end = offset + length - 1;
        const response = await fetch(this.source.remoteUrl, {
            method: 'GET',
            headers: { 'Range': `bytes=${offset}-${end}` }
        });
        if (response.status !== 206 && !response.ok) {
            throw new Error(`HTTP Range non supportato: ${response.status}`);
        }
        return await response.arrayBuffer();
    }
}
