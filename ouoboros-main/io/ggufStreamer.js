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
    
    // Genera l'indice topologico .ouro dal file GGUF usando parser completo
    async generateTopologyFromGguf() {
        const file = this.source.fileObject;
        
        try {
            // Leggi l'header GGUF completo
            const header = await this.readGgufHeaderComplete(file);
            console.log('[GGUF] Header parsed successfully:', {
                version: header.version,
                tensorCount: header.tensorCount,
                metadataKVCount: header.metadataKVCount,
                alignment: header.alignment
            });
            
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
    
    // Parser GGUF completo seguendo le specifiche ufficiali
    async readGgufHeaderComplete(file) {
        // Leggi i primi 128 byte per ottenere le informazioni base
        const initialBuffer = await this.readLocalSlice(0, 128);
        const initialView = new DataView(initialBuffer);
        
        // Check magic number "GGUF"
        const magic = initialView.getUint32(0, true);
        if (magic !== 0x46554747) {
            throw new Error(`Magic number GGUF non valido: 0x${magic.toString(16)}`);
        }
        
        const version = initialView.getUint32(4, true);
        const tensorCount = initialView.getUint32(8, true);
        const metadataKVCount = initialView.getUint32(12, true);
        
        // GGUF v3 ha alignment field
        let alignment = 32;
        if (version >= 3) {
            alignment = initialView.getUint32(16, true);
        }
        
        console.log(`[GGUF] Version: ${version}, Tensors: ${tensorCount}, Metadata: ${metadataKVCount}, Alignment: ${alignment}`);
        
        // Calcola dimensione header dinamica
        // Stima: 16-20 byte base + metadata (variabile) + tensor info (variabile)
        // Per sicurezza, leggi tutto fino a 500MB per modelli molto grandi
        const maxHeaderSize = 500 * 1024 * 1024; // 500MB max
        const headerBuffer = await this.readLocalSlice(0, maxHeaderSize);
        const view = new DataView(headerBuffer);
        
        let offset = version >= 3 ? 20 : 16;
        
        // Parse metadata key-value pairs
        const metadata = {};
        for (let i = 0; i < metadataKVCount; i++) {
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for metadata KV ${i}, stopping`);
                break;
            }
            
            const keyLen = view.getUint32(offset, true);
            offset += 4;
            
            if (offset + keyLen > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for key at KV ${i}, stopping`);
                break;
            }
            
            const keyBytes = new Uint8Array(headerBuffer, offset, keyLen);
            const key = new TextDecoder().decode(keyBytes);
            offset += keyLen;
            
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for type at KV ${i}, stopping`);
                break;
            }
            
            const type = view.getUint32(offset, true);
            offset += 4;
            
            try {
                const { value, newOffset } = this.readGgufValue(view, offset, type, headerBuffer.byteLength);
                metadata[key] = value;
                offset = newOffset;
            } catch (e) {
                console.warn(`[GGUF] Error reading value for key '${key}': ${e.message}`);
                break;
            }
            
            if (offset >= headerBuffer.byteLength) {
                console.warn(`[GGUF] Header exhausted at metadata KV ${i}, stopping`);
                break;
            }
        }
        
        console.log(`[GGUF] Metadata parsed, offset now: ${offset}`);
        
        // Parse tensor info
        const tensors = [];
        for (let i = 0; i < tensorCount; i++) {
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for tensor ${i}, stopping`);
                break;
            }
            
            const nameLen = view.getUint32(offset, true);
            offset += 4;
            
            if (offset + nameLen > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for tensor name at ${i}, stopping`);
                break;
            }
            
            const nameBytes = new Uint8Array(headerBuffer, offset, nameLen);
            const name = new TextDecoder().decode(nameBytes);
            offset += nameLen;
            
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for nDims at tensor ${i}, stopping`);
                break;
            }
            
            const nDims = view.getUint32(offset, true);
            offset += 4;
            
            const shape = [];
            for (let j = 0; j < nDims; j++) {
                if (offset + 4 > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Header too small for shape at tensor ${i}, stopping`);
                    break;
                }
                shape.push(view.getUint32(offset, true));
                offset += 4;
            }
            
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for dtype at tensor ${i}, stopping`);
                break;
            }
            
            const dtype = view.getUint32(offset, true);
            offset += 4;
            
            if (offset + 8 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for offset at tensor ${i}, stopping`);
                break;
            }
            
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
            
            // Log progress ogni 100 tensori
            if ((i + 1) % 100 === 0) {
                console.log(`[GGUF] Parsed ${i + 1}/${tensorCount} tensors`);
            }
        }
        
        console.log(`[GGUF] Total tensors parsed: ${tensors.length}/${tensorCount}`);
        
        return {
            version,
            tensorCount,
            metadataKVCount,
            alignment,
            metadata,
            tensors
        };
    }
    
    // Legge un valore GGUF con tipo specificato
    readGgufValue(view, offset, type, maxOffset) {
        switch (type) {
            case GGUF_TYPE.UINT8:
                return { value: view.getUint8(offset), newOffset: offset + 1 };
            case 1: // UINT8 alias
                return { value: view.getUint8(offset), newOffset: offset + 1 };
            case GGUF_TYPE.UINT16:
                return { value: view.getUint16(offset, true), newOffset: offset + 2 };
            case GGUF_TYPE.UINT32:
                return { value: view.getUint32(offset, true), newOffset: offset + 4 };
            case GGUF_TYPE.UINT64:
                const low = view.getUint32(offset, true);
                const high = view.getUint32(offset + 4, true);
                return { value: (BigInt(high) << 32n) | BigInt(low), newOffset: offset + 8 };
            case GGUF_TYPE.INT8:
                return { value: view.getInt8(offset), newOffset: offset + 1 };
            case 6: // INT8 alias
                return { value: view.getInt8(offset), newOffset: offset + 1 };
            case GGUF_TYPE.INT16:
                return { value: view.getInt16(offset, true), newOffset: offset + 2 };
            case GGUF_TYPE.INT32:
                return { value: view.getInt32(offset, true), newOffset: offset + 4 };
            case GGUF_TYPE.INT64:
                const intLow = view.getUint32(offset, true);
                const intHigh = view.getInt32(offset + 4, true);
                return { value: (BigInt(intHigh) << 32n) | BigInt(intLow), newOffset: offset + 8 };
            case GGUF_TYPE.FLOAT32:
                return { value: view.getFloat32(offset, true), newOffset: offset + 4 };
            case GGUF_TYPE.FLOAT64:
                return { value: view.getFloat64(offset, true), newOffset: offset + 8 };
            case GGUF_TYPE.BOOL:
                return { value: view.getUint8(offset) !== 0, newOffset: offset + 1 };
            case GGUF_TYPE.STRING:
                if (offset + 4 > maxOffset) throw new Error('String length out of bounds');
                const strLen = view.getUint32(offset, true);
                if (offset + 4 + strLen > maxOffset) throw new Error('String data out of bounds');
                const strBytes = new Uint8Array(view.buffer, offset + 4, strLen);
                const str = new TextDecoder().decode(strBytes);
                return { value: str, newOffset: offset + 4 + strLen };
            case GGUF_TYPE.ARRAY:
                if (offset + 8 > maxOffset) throw new Error('Array header out of bounds');
                const arrayLen = view.getUint32(offset, true);
                const arrayType = view.getUint32(offset + 4, true);
                offset += 8;
                const arrayValues = [];
                for (let i = 0; i < arrayLen; i++) {
                    const { value, newOffset } = this.readGgufValue(view, offset, arrayType, maxOffset);
                    arrayValues.push(value);
                    offset = newOffset;
                }
                return { value: arrayValues, newOffset: offset };
            default:
                console.warn(`[GGUF] Unknown type ${type}, skipping 4 bytes`);
                return { value: null, newOffset: offset + 4 };
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
