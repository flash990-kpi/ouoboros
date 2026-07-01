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
        // Prima leggi solo i primi 16 byte per ottenere le dimensioni
        const initialBuffer = await this.readLocalSlice(0, 16);
        const initialView = new DataView(initialBuffer);
        
        // Check magic number "GGUF"
        const magic = initialView.getUint32(0, true);
        if (magic !== 0x46554747) {
            throw new Error('Magic number GGUF non valido');
        }
        
        const version = initialView.getUint32(4, true);
        const tensorCount = initialView.getUint32(8, true);
        const metadataKVCount = initialView.getUint32(12, true);
        
        console.log(`[GGUF] Version: ${version}, Tensors: ${tensorCount}, Metadata: ${metadataKVCount}`);
        
        // Stima dimensione header: 16 byte base + metadata (stimato 1KB per KV) + tensor info (stimato 2KB per tensore per modelli grandi)
        const estimatedHeaderSize = 16 + (metadataKVCount * 1024) + (tensorCount * 2048);
        const maxHeaderSize = Math.min(estimatedHeaderSize, 200 * 1024 * 1024); // Max 200MB
        const headerSize = Math.max(maxHeaderSize, 50 * 1024 * 1024); // Minimo 50MB per modelli grandi
        
        console.log(`[GGUF] Reading header with buffer size: ${headerSize / 1024 / 1024}MB`);
        
        const headerBuffer = await this.readLocalSlice(0, headerSize);
        const view = new DataView(headerBuffer);
        
        // Debug: stampa i primi 100 byte in hex
        console.log('[GGUF] First 100 bytes hex dump:');
        for (let i = 0; i < 100 && i < headerBuffer.byteLength; i++) {
            const byte = headerBuffer[i].toString(16).padStart(2, '0');
            process.stdout.write(byte + ' ');
            if ((i + 1) % 16 === 0) console.log();
        }
        console.log();
        
        let offset = 16;
        
        // Skip metadata key-value pairs con controllo bounds
        for (let i = 0; i < metadataKVCount; i++) {
            console.log(`[GGUF] Processing metadata KV ${i}/${metadataKVCount} at offset ${offset}`);
            
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for metadata, stopping at ${i}/${metadataKVCount}`);
                break;
            }
            
            const keyLen = view.getUint32(offset, true);
            console.log(`[GGUF] Key length: ${keyLen}`);
            offset += 4;
            
            if (offset + keyLen > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for key, stopping at ${i}/${metadataKVCount}`);
                break;
            }
            const keyBytes = new Uint8Array(headerBuffer, offset, keyLen);
            const key = new TextDecoder().decode(keyBytes);
            console.log(`[GGUF] Key: ${key}`);
            offset += keyLen;
            
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for type, stopping at ${i}/${metadataKVCount}`);
                break;
            }
            const type = view.getUint32(offset, true);
            console.log(`[GGUF] Type: ${type}`);
            offset += 4;
            
            // Skip value based on type
            try {
                const newOffset = this.skipGgufValue(view, offset, type, headerBuffer.byteLength);
                console.log(`[GGUF] Skipped value, offset moved from ${offset} to ${newOffset}`);
                offset = newOffset;
            } catch (e) {
                console.warn(`[GGUF] Error skipping value type ${type}: ${e.message}`);
                break;
            }
            
            if (offset >= headerBuffer.byteLength) {
                console.warn(`[GGUF] Header exhausted, stopping at ${i}/${metadataKVCount}`);
                break;
            }
        }
        
        console.log(`[GGUF] After metadata, offset is ${offset}`);
        
        // Read tensor info con controllo bounds
        const tensors = [];
        for (let i = 0; i < tensorCount; i++) {
            console.log(`[GGUF] Processing tensor ${i}/${tensorCount} at offset ${offset}`);
            
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for tensor info, stopping at ${i}/${tensorCount}`);
                break;
            }
            
            const nameLen = view.getUint32(offset, true);
            console.log(`[GGUF] Tensor name length: ${nameLen}`);
            offset += 4;
            
            if (offset + nameLen > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for tensor name, stopping at ${i}/${tensorCount}`);
                break;
            }
            const nameBytes = new Uint8Array(headerBuffer, offset, nameLen);
            const name = new TextDecoder().decode(nameBytes);
            console.log(`[GGUF] Tensor name: ${name}`);
            offset += nameLen;
            
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for nDims, stopping at ${i}/${tensorCount}`);
                break;
            }
            const nDims = view.getUint32(offset, true);
            console.log(`[GGUF] Tensor nDims: ${nDims}`);
            offset += 4;
            
            const shape = [];
            for (let j = 0; j < nDims; j++) {
                if (offset + 4 > headerBuffer.byteLength) {
                    console.warn(`[GGUF] Header too small for shape, stopping at ${i}/${tensorCount}`);
                    break;
                }
                const dim = view.getUint32(offset, true);
                shape.push(dim);
                console.log(`[GGUF] Shape[${j}]: ${dim}`);
                offset += 4;
            }
            
            if (offset + 4 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for dtype, stopping at ${i}/${tensorCount}`);
                break;
            }
            const dtype = view.getUint32(offset, true);
            console.log(`[GGUF] Tensor dtype: ${dtype}`);
            offset += 4;
            
            if (offset + 8 > headerBuffer.byteLength) {
                console.warn(`[GGUF] Header too small for offset, stopping at ${i}/${tensorCount}`);
                break;
            }
            const offsetLow = view.getUint32(offset, true);
            offset += 4;
            const offsetHigh = view.getUint32(offset, true);
            offset += 4;
            const tensorOffset = (BigInt(offsetHigh) << 32n) | BigInt(offsetLow);
            console.log(`[GGUF] Tensor offset: ${tensorOffset}`);
            
            tensors.push({
                name,
                shape,
                dtype,
                offset: tensorOffset
            });
            
            // Log solo i primi 5 tensori per non floodare la console
            if (i >= 4) {
                console.log(`[GGUF] Skipping logging for remaining tensors...`);
            }
        }
        
        console.log(`[GGUF] Parsed ${tensors.length} tensors`);
        
        return {
            version,
            tensorCount,
            tensors
        };
    }
    
    skipGgufValue(view, offset, type, maxOffset) {
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
                if (offset + 4 > maxOffset) throw new Error('String length out of bounds');
                const len = view.getUint32(offset, true);
                return offset + 4 + len;
            case 14: // ARRAY
                if (offset + 8 > maxOffset) throw new Error('Array header out of bounds');
                const arrayLen = view.getUint32(offset, true);
                offset += 4;
                const arrayType = view.getUint32(offset, true);
                offset += 4;
                for (let i = 0; i < arrayLen; i++) {
                    offset = this.skipGgufValue(view, offset, arrayType, maxOffset);
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
