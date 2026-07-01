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
    
    // Genera l'indice topologico .ouro dal file GGUF usando sistema multi-parser
    async generateTopologyFromGguf() {
        const file = this.source.fileObject;
        
        try {
            console.log('[GGUF] Starting multi-parser approach...');
            
            // Strategia 1: Usa @huggingface/transformers se disponibile
            let header = null;
            try {
                header = await this.parseWithTransformers(file);
                console.log('[GGUF] Transformers parser succeeded');
            } catch (e) {
                console.warn('[GGUF] Transformers parser failed:', e.message);
            }
            
            // Strategia 2: Usa parser personalizzato se Transformers fallisce
            if (!header || header.tensors.length === 0) {
                console.log('[GGUF] Falling back to custom parser...');
                header = await this.readGgufHeaderComplete(file);
            }
            
            if (!header || header.tensors.length === 0) {
                throw new Error('Nessun parser è riuscito a leggere i tensori dal file GGUF');
            }
            
            console.log('[GGUF] Header parsed successfully:', {
                version: header.version,
                tensorCount: header.tensorCount,
                metadataKVCount: header.metadataKVCount,
                alignment: header.alignment,
                actualTensors: header.tensors.length
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
    
    // Parser usando @huggingface/gguf (se disponibile)
    async parseWithTransformers(file) {
        try {
            // Tenta di importare dinamicamente la libreria
            const ggufLib = await import('@huggingface/gguf');
            
            // Crea un GGUF reader per leggere l'header
            const buffer = await file.arrayBuffer();
            const uint8Array = new Uint8Array(buffer);
            
            // Usa il parser GGUF
            const gguf = await ggufLib.GGUF.from_buffer(uint8Array);
            
            const tensors = [];
            for (const tensor of gguf.tensors) {
                tensors.push({
                    name: tensor.name,
                    shape: tensor.shape,
                    dtype: tensor.dtype,
                    offset: BigInt(tensor.data_offset)
                });
            }
            
            return {
                version: gguf.version,
                tensorCount: gguf.tensor_count,
                metadataKVCount: gguf.metadata_kv_count,
                alignment: gguf.alignment || 32,
                tensors
            };
        } catch (e) {
            throw new Error(`GGUF parser non disponibile o fallito: ${e.message}`);
        }
    }
    
    // Parser GGUF minimalista per file molto grandi o non standard
    async readGgufHeaderComplete(file) {
        console.log('[GGUF] Starting minimal parser for large files...');
        
        // Leggi i primi 256 byte per debugging
        const debugBuffer = await this.readLocalSlice(0, 256);
        const debugView = new DataView(debugBuffer);
        
        // Debug: stampa i primi 64 byte in hex
        console.log('[GGUF] First 64 bytes hex:');
        let hexString = '';
        for (let i = 0; i < 64 && i < debugBuffer.byteLength; i++) {
            hexString += debugBuffer[i].toString(16).padStart(2, '0') + ' ';
        }
        console.log(hexString);
        
        // Check magic number "GGUF"
        const magic = debugView.getUint32(0, true);
        console.log(`[GGUF] Magic: 0x${magic.toString(16)} (expected: 0x46554747)`);
        if (magic !== 0x46554747) {
            throw new Error(`Magic number GGUF non valido: 0x${magic.toString(16)}. Il file potrebbe non essere un GGUF valido.`);
        }
        
        const version = debugView.getUint32(4, true);
        const tensorCount = debugView.getUint32(8, true);
        const metadataKVCount = debugView.getUint32(12, true);
        
        console.log(`[GGUF] Version: ${version}, Tensors: ${tensorCount}, Metadata: ${metadataKVCount}`);
        
        // GGUF v3 ha alignment field
        let alignment = 32;
        let headerStartOffset = 16;
        if (version >= 3) {
            alignment = debugView.getUint32(16, true);
            headerStartOffset = 20;
            console.log(`[GGUF] Alignment: ${alignment}`);
        }
        
        // Per file molto grandi, usa una strategia diversa:
        // Leggi solo l'offset del primo tensore e calcola la dimensione stimata
        const tensorInfoOffset = headerStartOffset;
        
        // Calcola dimensione header stimata basata sul numero di tensori
        // Ogni tensore ha: nameLen (4) + name + nDims (4) + dims (4*nDims) + dtype (4) + offset (8)
        // Stimiamo 50 byte per tensore in media
        const estimatedHeaderSize = tensorInfoOffset + (metadataKVCount * 100) + (tensorCount * 100);
        const maxHeaderSize = Math.min(estimatedHeaderSize, 200 * 1024 * 1024); // Max 200MB
        
        console.log(`[GGUF] Estimated header size: ${maxHeaderSize / 1024 / 1024}MB`);
        
        const headerBuffer = await this.readLocalSlice(0, maxHeaderSize);
        const view = new DataView(headerBuffer);
        
        let offset = headerStartOffset;
        
        // Skip metadata (per ora, non ci serve per l'indice topologico)
        console.log(`[GGUF] Skipping ${metadataKVCount} metadata entries...`);
        for (let i = 0; i < metadataKVCount; i++) {
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Cannot skip metadata KV ${i}, breaking`);
                break;
            }
            const keyLen = view.getUint32(offset, true);
            offset += 4;
            
            if (offset + keyLen > headerBuffer.byteLength) {
                console.warn(`[GGUF] Cannot skip key at KV ${i}, breaking`);
                break;
            }
            offset += keyLen;
            
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Cannot skip type at KV ${i}, breaking`);
                break;
            }
            const type = view.getUint32(offset, true);
            offset += 4;
            
            // Skip value in modo sicuro
            try {
                offset = this.skipGgufValueSafe(view, offset, type, headerBuffer.byteLength);
            } catch (e) {
                console.warn(`[GGUF] Error skipping value type ${type}: ${e.message}, breaking`);
                break;
            }
        }
        
        console.log(`[GGUF] After metadata, offset: ${offset}`);
        
        // Parse tensor info con estrema cautela
        const tensors = [];
        for (let i = 0; i < tensorCount; i++) {
            try {
                if (offset + 4 > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Cannot read tensor nameLen at ${i}, stopping`);
                    break;
                }
                
                const nameLen = view.getUint32(offset, true);
                offset += 4;
                
                // Sanity check: nameLen non dovrebbe essere > 1000
                if (nameLen > 10000) {
                    console.error(`[GGUF] Invalid nameLen ${nameLen} at tensor ${i}, file may be corrupted`);
                    throw new Error(`Invalid tensor name length: ${nameLen}. File may be corrupted.`);
                }
                
                if (offset + nameLen > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Cannot read tensor name at ${i}, stopping`);
                    break;
                }
                
                const nameBytes = new Uint8Array(headerBuffer, offset, nameLen);
                const name = new TextDecoder().decode(nameBytes);
                offset += nameLen;
                
                if (offset + 4 > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Cannot read nDims at tensor ${i}, stopping`);
                    break;
                }
                
                const nDims = view.getUint32(offset, true);
                offset += 4;
                
                // Sanity check: nDims non dovrebbe essere > 10
                if (nDims > 10) {
                    console.error(`[GGUF] Invalid nDims ${nDims} at tensor ${i}, file may be corrupted`);
                    throw new Error(`Invalid tensor dimensions: ${nDims}. File may be corrupted.`);
                }
                
                const shape = [];
                for (let j = 0; j < nDims; j++) {
                    if (offset + 4 > headerBuffer.byteLength) {
                        console.warn(`[GGUF] Cannot read shape[${j}] at tensor ${i}, stopping`);
                        break;
                    }
                    shape.push(view.getUint32(offset, true));
                    offset += 4;
                }
                
                if (offset + 4 > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Cannot read dtype at tensor ${i}, stopping`);
                    break;
                }
                
                const dtype = view.getUint32(offset, true);
                offset += 4;
                
                if (offset + 8 > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Cannot read offset at tensor ${i}, stopping`);
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
                
                if ((i + 1) % 100 === 0) {
                    console.log(`[GGUF] Parsed ${i + 1}/${tensorCount} tensors`);
                }
            } catch (e) {
                console.error(`[GGUF] Error parsing tensor ${i}: ${e.message}`);
                throw e;
            }
        }
        
        console.log(`[GGUF] Successfully parsed ${tensors.length}/${tensorCount} tensors`);
        
        if (tensors.length === 0) {
            throw new Error('Nessun tensore parsato. Il file potrebbe essere corrotto o in un formato non supportato.');
        }
        
        return {
            version,
            tensorCount,
            metadataKVCount,
            alignment,
            metadata: {},
            tensors
        };
    }
    
    // Skip GGUF value in modo sicuro
    skipGgufValueSafe(view, offset, type, maxOffset) {
        switch (type) {
            case 0: // UINT8
            case 1: // UINT8
                return offset + 1;
            case 2: // UINT16
                return offset + 2;
            case 3: // UINT32
                return offset + 4;
            case 4: // UINT64
                return offset + 8;
            case 5: // INT8
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
                if (offset + 4 > maxOffset) throw new Error('String length out of bounds');
                const len = view.getUint32(offset, true);
                if (len > 1000000) throw new Error(`String too long: ${len}`);
                return offset + 4 + len;
            case 14: // ARRAY
                if (offset + 8 > maxOffset) throw new Error('Array header out of bounds');
                const arrayLen = view.getUint32(offset, true);
                if (arrayLen > 1000000) throw new Error(`Array too long: ${arrayLen}`);
                offset += 4;
                const arrayType = view.getUint32(offset, true);
                offset += 4;
                for (let i = 0; i < arrayLen; i++) {
                    offset = this.skipGgufValueSafe(view, offset, arrayType, maxOffset);
                }
                return offset;
            default:
                console.warn(`[GGUF] Unknown type ${type}, skipping 4 bytes`);
                return offset + 4;
        }
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
