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
    
    // Genera l'indice topologico .ouro dal file GGUF usando Web Streams
    async generateTopologyFromGguf() {
        const file = this.source.fileObject;
        
        try {
            console.log('[GGUF] Using Web Streams for large file support...');
            console.log(`[GGUF] File size: ${file.size} bytes`);
            
            // Leggi i primi 256 byte per il header
            console.log('[GGUF] Reading first 256 bytes...');
            const headerBytes = await this.readStreamChunk(0, 256);
            console.log(`[GGUF] Read ${headerBytes.byteLength} bytes`);
            
            const headerView = new DataView(headerBytes);
            const headerArray = new Uint8Array(headerBytes);
            
            // Debug hex dump
            console.log('[GGUF] First 32 bytes hex:');
            let hex = '';
            for (let i = 0; i < 32; i++) {
                hex += headerArray[i].toString(16).padStart(2, '0') + ' ';
            }
            console.log(hex);
            
            // Check magic number
            const magic = headerView.getUint32(0, true);
            console.log(`[GGUF] Magic: 0x${magic.toString(16)}`);
            if (magic !== 0x46554747) {
                throw new Error(`Magic number GGUF non valido: 0x${magic.toString(16)}`);
            }
            
            const version = headerView.getUint32(4, true);
            const tensorCount = Number(headerView.getBigUint64(8, true));
            const metadataKVCount = Number(headerView.getBigUint64(16, true));
            
            console.log(`[GGUF] Version: ${version}, Tensors: ${tensorCount}, Metadata: ${metadataKVCount}`);
            
            // GGUF v3 ha alignment field
            let alignment = 32;
            let headerStartOffset = 24;
            if (version >= 3) {
                alignment = headerView.getUint32(24, true);
                headerStartOffset = 28;
                console.log(`[GGUF] Alignment: ${alignment}`);
            }
            
            // Calcola dimensione header stimata
            const estimatedHeaderSize = headerStartOffset + (metadataKVCount * 100) + (tensorCount * 100);
            const maxHeaderSize = Math.min(estimatedHeaderSize, 50 * 1024 * 1024); // Max 50MB
            
            console.log(`[GGUF] Reading header: ${maxHeaderSize / 1024 / 1024}MB`);
            const headerBuffer = await this.readStreamChunk(0, maxHeaderSize);
            console.log(`[GGUF] Header buffer size: ${headerBuffer.byteLength} bytes`);
            
            const view = new DataView(headerBuffer);
            const headerArrayFull = new Uint8Array(headerBuffer);
            
            let offset = headerStartOffset;
            console.log(`[GGUF] Starting offset: ${offset}`);
            
            // Skip metadata
            console.log(`[GGUF] Skipping ${metadataKVCount} metadata entries...`);
            for (let i = 0; i < metadataKVCount; i++) {
                if (offset + 8 > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Break at metadata ${i}: offset ${offset} > ${headerBuffer.byteLength}`);
                    break;
                }
                const keyLen = Number(view.getBigUint64(offset, true));
                offset += 8 + keyLen + 4; // keyLen + key + type
                const type = view.getUint32(offset - 4, true);
                offset = this.skipValue(view, offset, type, headerBuffer.byteLength);
            }
            console.log(`[GGUF] After metadata, offset: ${offset}`);
            
            // Debug: mostra 32 byte dopo metadata
            console.log('[GGUF] 32 bytes after metadata:');
            let debugHex = '';
            for (let i = 0; i < 32 && offset + i < headerArrayFull.length; i++) {
                debugHex += headerArrayFull[offset + i].toString(16).padStart(2, '0') + ' ';
            }
            console.log(debugHex);
            
            // Parse tensors
            const tensors = [];
            for (let i = 0; i < tensorCount; i++) {
                if (offset + 8 > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Break at tensor ${i}: offset ${offset} > ${headerBuffer.byteLength}`);
                    break;
                }
                
                // nameLen: 8 byte (uint64_t)
                const nameLen = Number(view.getBigUint64(offset, true));
                console.log(`[GGUF] Tensor ${i}: nameLen=${nameLen} at offset=${offset}`);
                offset += 8;
                
                if (offset + nameLen > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Break at tensor ${i}: nameLen ${nameLen} exceeds buffer`);
                    break;
                }
                const name = new TextDecoder().decode(headerArrayFull.slice(offset, offset + nameLen));
                console.log(`[GGUF] Tensor ${i}: name="${name}"`);
                offset += nameLen;
                
                if (offset + 4 > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Break at tensor ${i}: cannot read nDims`);
                    break;
                }
                // nDims: 4 byte (uint32_t)
                const nDims = view.getUint32(offset, true);
                console.log(`[GGUF] Tensor ${i}: nDims=${nDims}`);
                offset += 4;
                
                const shape = [];
                for (let j = 0; j < nDims; j++) {
                    if (offset + 8 > headerBuffer.byteLength) {
                        console.warn(`[GGUF] Break at tensor ${i}: cannot read shape[${j}]`);
                        break;
                    }
                    // dimensions: 8 byte (uint64_t) per dimensione
                    shape.push(view.getBigUint64(offset, true));
                    offset += 8;
                }
                
                if (offset + 4 > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Break at tensor ${i}: cannot read type`);
                    break;
                }
                // type: 4 byte (uint32_t)
                const dtype = view.getUint32(offset, true);
                offset += 4;
                
                if (offset + 8 > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Break at tensor ${i}: cannot read offset`);
                    break;
                }
                // offset: 8 byte (uint64_t)
                const tensorOffset = view.getBigUint64(offset, true);
                offset += 8;
                
                tensors.push({ name, shape, dtype, offset: tensorOffset });
                console.log(`[GGUF] Tensor ${i} parsed: offset=${tensorOffset}, shape=[${shape.join(',')}]`);
                
                if ((i + 1) % 100 === 0) {
                    console.log(`[GGUF] Parsed ${i + 1}/${tensorCount} tensors`);
                }
            }
            
            console.log(`[GGUF] Successfully parsed ${tensors.length}/${tensorCount} tensors`);
            
            if (tensors.length === 0) {
                throw new Error('Nessun tensore parsato. Header buffer potrebbe essere troppo piccolo.');
            }
            
            // Genera .ouro
            const tensorCountParsed = tensors.length;
            let maxLayerFound = 0;
            
            const ouroBuffer = new ArrayBuffer(32 + (tensorCountParsed * 48));
            const ouroView = new DataView(ouroBuffer);
            
            ouroView.setUint32(0, 0x4f55524f, true);
            ouroView.setUint32(4, 1, true);
            ouroView.setUint32(8, tensorCountParsed, true);
            
            let ouroCursor = 32;
            
            for (const tensor of tensors) {
                const name = tensor.name;
                const offset = tensor.offset;
                const shape = tensor.shape;
                const dtype = tensor.dtype;
                
                let totalElements = 1;
                for (const dim of shape) totalElements *= Number(dim);
                
                const dtypeMap = { 0: 4, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 1, 15: 2, 16: 4, 17: 8 };
                const dtypeNumber = typeof dtype === 'number' ? dtype : 0;
                const elementSize = dtypeMap[dtypeNumber] || 4;
                const byteLength = BigInt(totalElements * elementSize);
                
                const layerMatch = name.match(/blk\.(\d+)\./) || name.match(/layers\.(\d+)\./) || name.match(/layer\.(\d+)\./);
                const layerIndex = layerMatch ? parseInt(layerMatch[1], 10) : 0;
                if (layerIndex > maxLayerFound) maxLayerFound = layerIndex;
                
                let sparsityRank = 4;
                if (name.includes("ffn_down") || name.includes("v_proj") || name.includes("down_proj"))
                    sparsityRank = 1;
                else if (name.includes("attn_q") || name.includes("attn_k") || name.includes("q_proj") || name.includes("k_proj"))
                    sparsityRank = 2;
                else if (name.includes("ffn_up") || name.includes("ffn_gate") || name.includes("up_proj") || name.includes("gate_proj"))
                    sparsityRank = 3;
                
                const tensorHash = TopologyParser.hashTensorName(name);
                
                ouroView.setBigUint64(ouroCursor, tensorHash, true);
                ouroView.setBigUint64(ouroCursor + 8, offset, true);
                ouroView.setBigUint64(ouroCursor + 16, byteLength, true);
                ouroView.setUint32(ouroCursor + 24, layerIndex, true);
                ouroView.setUint32(ouroCursor + 28, dtypeNumber, true);
                ouroView.setUint32(ouroCursor + 32, sparsityRank, true);
                ouroView.setBigUint64(ouroCursor + 36, 0n, true);
                ouroCursor += 48;
            }
            
            ouroView.setUint32(12, maxLayerFound + 1, true);
            
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
    
    // Leggi chunk usando Web Streams API
    async readStreamChunk(offset, length) {
        const file = this.source.fileObject;
        console.log(`[GGUF] Reading stream chunk: offset=${offset}, length=${length}`);
        
        try {
            const stream = file.stream();
            const reader = stream.getReader();
            
            // Skip bytes prima dell'offset
            let bytesSkipped = 0;
            while (bytesSkipped < offset) {
                const { done, value } = await reader.read();
                if (done) break;
                bytesSkipped += value.length;
            }
            
            // Leggi i byte richiesti
            const chunks = [];
            let bytesRead = 0;
            
            while (bytesRead < length) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const remaining = length - bytesRead;
                if (value.length <= remaining) {
                    chunks.push(value);
                    bytesRead += value.length;
                } else {
                    chunks.push(value.slice(0, remaining));
                    bytesRead += remaining;
                }
            }
            
            reader.releaseLock();
            
            const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
            const result = new Uint8Array(totalLength);
            let position = 0;
            
            for (const chunk of chunks) {
                result.set(chunk, position);
                position += chunk.length;
            }
            
            console.log(`[GGUF] Read ${totalLength} bytes via streams`);
            return result.buffer;
        } catch (e) {
            console.error('[GGUF] Stream error:', e);
            // Fallback a FileReader
            return this.readLocalSlice(offset, length);
        }
    }
    
    // Skip GGUF value
    skipValue(view, offset, type, maxOffset) {
        switch (type) {
            case 0: case 1: return offset + 1;
            case 2: return offset + 2;
            case 3: return offset + 4;
            case 4: return offset + 8;
            case 5: case 6: return offset + 1;
            case 7: return offset + 2;
            case 8: return offset + 4;
            case 9: return offset + 8;
            case 10: return offset + 4;
            case 11: return offset + 8;
            case 12: return offset + 1;
            case 13:
                if (offset + 4 > maxOffset) return offset;
                const len = view.getUint32(offset, true);
                return offset + 4 + len;
            case 14:
                if (offset + 8 > maxOffset) return offset;
                const arrayLen = view.getUint32(offset, true);
                offset += 4;
                const arrayType = view.getUint32(offset, true);
                offset += 4;
                for (let i = 0; i < arrayLen; i++) {
                    offset = this.skipValue(view, offset, arrayType, maxOffset);
                }
                return offset;
            default: return offset + 4;
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
