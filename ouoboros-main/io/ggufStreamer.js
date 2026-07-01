import { TopologyParser } from '../asts/topologyParser.js';
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
    // Genera l'indice topologico .ouro dal file GGUF usando parser personalizzato
    async generateTopologyFromGguf() {
        const file = this.source.fileObject;
        
        try {
            // Leggi l'header GGUF
            const header = await this.readGgufHeader(file);
            console.log('[GGUF] Header parsed:', header);
            
            // Genera l'indice topologico
            const tensorCount = header.tensorCount;
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
            
            for (const tensor of header.tensors) {
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
                
                // Estrai layer index dal nome (es. "blk.0.attn_q.weight")
                const layerMatch = name.match(/blk\.(\d+)\./);
                const layerIndex = layerMatch ? parseInt(layerMatch[1], 10) : 0;
                if (layerIndex > maxLayerFound)
                    maxLayerFound = layerIndex;
                
                // Sparsity rank (euristica)
                let sparsityRank = 4;
                if (name.includes("ffn_down") || name.includes("v_proj"))
                    sparsityRank = 1;
                else if (name.includes("attn_q") || name.includes("attn_k"))
                    sparsityRank = 2;
                else if (name.includes("ffn_up") || name.includes("ffn_gate"))
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
    
    // Parser personalizzato per header GGUF
    async readGgufHeader(file) {
        const headerSize = 1024 * 1024; // 1MB per header
        const headerBuffer = await this.readLocalSlice(0, headerSize);
        const view = new DataView(headerBuffer);
        
        // Check magic number "GGUF"
        const magic = view.getUint32(0, true);
        if (magic !== 0x46554747) {
            throw new Error('Magic number GGUF non valido');
        }
        
        const version = view.getUint32(4, true);
        const tensorCount = view.getUint32(8, true);
        const metadataKVCount = view.getUint32(12, true);
        
        console.log(`[GGUF] Version: ${version}, Tensors: ${tensorCount}, Metadata: ${metadataKVCount}`);
        
        let offset = 16;
        
        // Skip metadata key-value pairs
        for (let i = 0; i < metadataKVCount; i++) {
            const keyLen = view.getUint32(offset, true);
            offset += 4;
            const keyBytes = new Uint8Array(headerBuffer, offset, keyLen);
            offset += keyLen;
            
            const type = view.getUint32(offset, true);
            offset += 4;
            
            // Skip value based on type
            offset = this.skipGgufValue(view, offset, type);
        }
        
        // Read tensor info
        const tensors = [];
        for (let i = 0; i < tensorCount; i++) {
            const nameLen = view.getUint32(offset, true);
            offset += 4;
            const nameBytes = new Uint8Array(headerBuffer, offset, nameLen);
            const name = new TextDecoder().decode(nameBytes);
            offset += nameLen;
            
            const nDims = view.getUint32(offset, true);
            offset += 4;
            
            const shape = [];
            for (let j = 0; j < nDims; j++) {
                shape.push(view.getUint32(offset, true));
                offset += 4;
            }
            
            const dtype = view.getUint32(offset, true);
            offset += 4;
            
            const offsetLow = view.getUint32(offset, true);
            offset += 4;
            const offsetHigh = view.getUint32(offset, true);
            offset += 4;
            const tensorOffset = (BigInt(offsetHigh) << 32n) | BigInt(offsetLow);
            
            tensors.push({
                name,
                shape,
                dtype,
                offset: tensorOffset
            });
        }
        
        return {
            version,
            tensorCount,
            tensors
        };
    }
    
    skipGgufValue(view, offset, type) {
        switch (type) {
            case 0: // UINT8
                return offset + 1;
            case 1: // UINT8
                return offset + 1;
            case 2: // UINT16
                return offset + 2;
            case 3: // UINT32
                return offset + 4;
            case 4: // UINT64
                return offset + 8;
            case 5: // INT8
                return offset + 1;
            case 6: // INT8
                return offset + 1;
            case 7: // INT16
                return offset + 2;
            case 8: // INT32
                return offset + 4;
            case 9: // INT64
                return offset + 8;
            case 10: // FLOAT32
                return offset + 4;
            case 11: // FLOAT64
                return offset + 8;
            case 12: // BOOL
                return offset + 1;
            case 13: // STRING
                const len = view.getUint32(offset, true);
                return offset + 4 + len;
            case 14: // ARRAY
                const arrayLen = view.getUint32(offset, true);
                offset += 4;
                const arrayType = view.getUint32(offset, true);
                offset += 4;
                for (let i = 0; i < arrayLen; i++) {
                    offset = this.skipGgufValue(view, offset, arrayType);
                }
                return offset;
            default:
                console.warn(`[GGUF] Unknown type ${type}, skipping 4 bytes`);
                return offset + 4;
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
            const slice = file.slice(offset, offset + length);
            const reader = new FileReader();
            reader.onload = () => {
                if (reader.result instanceof ArrayBuffer)
                    resolve(reader.result);
                else
                    reject(new Error("Errore lettura chunk."));
            };
            reader.onerror = () => reject(reader.error);
            reader.readAsArrayBuffer(slice);
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
