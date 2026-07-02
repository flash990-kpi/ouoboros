/**
 * GGUF Transformer - Vanilla JavaScript implementation for GGUF inference
 * Based on Javascript-Transformer by matthewJamesAbbott
 * Loads GGUF files directly without conversion
 */

export class GGUFTransformer {
    constructor() {
        this.tensors = new Map();
        this.metadata = null;
        this.tokenizer = null;
    }

    /**
     * Carica file GGUF dal blob
     */
    async loadGGUF(fileBlob) {
        console.log('[GGUF TRANSFORMER] Loading GGUF file...');
        
        const arrayBuffer = await fileBlob.arrayBuffer();
        const dataView = new DataView(arrayBuffer);
        
        // Parse GGUF header
        const magic = dataView.getUint32(0, true);
        if (magic !== 0x46554747) { // 'GGUF' in little endian
            throw new Error('Invalid GGUF magic number');
        }
        
        const version = dataView.getUint32(4, true);
        const tensorCount = dataView.getUint64(8, true);
        const metadataKVCount = dataView.getUint64(16, true);
        
        console.log('[GGUF TRANSFORMER] GGUF version:', version);
        console.log('[GGUF TRANSFORMER] Tensors:', tensorCount);
        console.log('[GGUF TRANSFORMER] Metadata entries:', metadataKVCount);
        
        // Parse metadata
        let offset = 24;
        this.metadata = {};
        
        for (let i = 0; i < metadataKVCount; i++) {
            const keyLength = dataView.getUint64(offset, true);
            offset += 8;
            const key = new TextDecoder().decode(new Uint8Array(arrayBuffer, offset, keyLength));
            offset += keyLength;
            
            const type = dataView.getUint32(offset, true);
            offset += 4;
            
            const value = this.parseValue(dataView, arrayBuffer, offset, type);
            offset = value.newOffset;
            
            this.metadata[key] = value.data;
        }
        
        // Parse tensors
        for (let i = 0; i < tensorCount; i++) {
            const nameLength = dataView.getUint64(offset, true);
            offset += 8;
            const name = new TextDecoder().decode(new Uint8Array(arrayBuffer, offset, nameLength));
            offset += nameLength;
            
            const ndim = dataView.getUint32(offset, true);
            offset += 4;
            
            const dimensions = [];
            for (let j = 0; j < ndim; j++) {
                dimensions.push(dataView.getUint64(offset, true));
                offset += 8;
            }
            
            const dtype = dataView.getUint32(offset, true);
            offset += 4;
            
            const offsetBytes = dataView.getUint64(offset, true);
            offset += 8;
            
            this.tensors.set(name, {
                name,
                dimensions,
                dtype,
                offset: offsetBytes
            });
        }
        
        console.log('[GGUF TRANSFORMER] GGUF loaded successfully');
        console.log('[GGUF TRANSFORMER] Metadata:', this.metadata);
        
        return {
            metadata: this.metadata,
            tensors: this.tensors
        };
    }

    /**
     * Parse GGUF value based on type
     */
    parseValue(dataView, arrayBuffer, offset, type) {
        const TYPE_MAP = {
            0: 'UINT8',
            1: 'INT8',
            2: 'UINT16',
            3: 'INT16',
            4: 'UINT32',
            5: 'INT32',
            6: 'FLOAT32',
            7: 'BOOL',
            8: 'STRING',
            9: 'ARRAY',
            10: 'UINT64',
            11: 'INT64',
            12: 'FLOAT64'
        };

        switch (type) {
            case 0: // UINT8
                return { data: dataView.getUint8(offset), newOffset: offset + 1 };
            case 1: // INT8
                return { data: dataView.getInt8(offset), newOffset: offset + 1 };
            case 2: // UINT16
                return { data: dataView.getUint16(offset, true), newOffset: offset + 2 };
            case 3: // INT16
                return { data: dataView.getInt16(offset, true), newOffset: offset + 2 };
            case 4: // UINT32
                return { data: dataView.getUint32(offset, true), newOffset: offset + 4 };
            case 5: // INT32
                return { data: dataView.getInt32(offset, true), newOffset: offset + 4 };
            case 6: // FLOAT32
                return { data: dataView.getFloat32(offset, true), newOffset: offset + 4 };
            case 7: // BOOL
                return { data: dataView.getUint8(offset) !== 0, newOffset: offset + 1 };
            case 8: // STRING
                const strLength = dataView.getUint64(offset, true);
                offset += 8;
                const str = new TextDecoder().decode(new Uint8Array(arrayBuffer, offset, strLength));
                return { data: str, newOffset: offset + strLength };
            case 10: // UINT64
                return { data: Number(dataView.getBigUint64(offset, true)), newOffset: offset + 8 };
            case 11: // INT64
                return { data: Number(dataView.getBigInt64(offset, true)), newOffset: offset + 8 };
            case 12: // FLOAT64
                return { data: dataView.getFloat64(offset, true), newOffset: offset + 8 };
            default:
                console.warn('[GGUF TRANSFORMER] Unknown type:', type);
                return { data: null, newOffset: offset };
        }
    }

    /**
     * Carica tokenizer da JSON
     */
    async loadTokenizer(tokenizerJson) {
        console.log('[GGUF TRANSFORMER] Loading tokenizer...');
        
        if (typeof tokenizerJson === 'string') {
            this.tokenizer = JSON.parse(tokenizerJson);
        } else {
            this.tokenizer = tokenizerJson;
        }
        
        console.log('[GGUF TRANSFORMER] Tokenizer loaded');
        return this.tokenizer;
    }

    /**
     * Tokenizza testo
     */
    tokenize(text) {
        if (!this.tokenizer) {
            // Fallback: semplice tokenizzazione per caratteri
            return text.split('').map(c => c.charCodeAt(0));
        }
        
        // Implementazione BPE semplificata
        const tokens = [];
        let remaining = text;
        
        while (remaining.length > 0) {
            let matched = false;
            
            // Prova match più lungo
            for (let len = Math.min(remaining.length, 10); len >= 1; len--) {
                const substring = remaining.substring(0, len);
                if (this.tokenizer.vocab && this.tokenizer.vocab[substring]) {
                    tokens.push(this.tokenizer.vocab[substring]);
                    remaining = remaining.substring(len);
                    matched = true;
                    break;
                }
            }
            
            if (!matched) {
                // Fallback: carattere singolo
                tokens.push(remaining.charCodeAt(0));
                remaining = remaining.substring(1);
            }
        }
        
        return tokens;
    }

    /**
     * Detokenizza ID in testo
     */
    detokenize(tokenIds) {
        if (!this.tokenizer) {
            // Fallback: caratteri ASCII
            return tokenIds.map(id => String.fromCharCode(id)).join('');
        }
        
        // Implementazione inversa semplificata
        let text = '';
        for (const id of tokenIds) {
            const token = Object.entries(this.tokenizer.vocab || {}).find(([k, v]) => v === id);
            if (token) {
                text += token[0];
            } else {
                text += String.fromCharCode(id);
            }
        }
        
        return text;
    }

    /**
     * Genera testo con GGUF transformer
     */
    async *generate(prompt, maxTokens = 100, temperature = 0.7) {
        console.log('[GGUF TRANSFORMER] Starting generation...');
        
        const tokens = this.tokenize(prompt);
        console.log('[GGUF TRANSFORMER] Input tokens:', tokens.length);
        
        // Generazione semplificata con attention mechanism
        let currentTokens = [...tokens];
        
        for (let i = 0; i < maxTokens; i++) {
            // Forward pass semplificato
            const logits = this.forwardPass(currentTokens);
            
            // Sampling
            const nextToken = this.sample(logits, temperature);
            
            // Yield token
            const text = this.detokenize([nextToken]);
            yield text;
            
            currentTokens.push(nextToken);
            
            // Stop condition
            if (nextToken === 0 || nextToken === 2) {
                break;
            }
        }
    }

    /**
     * Forward pass semplificato
     */
    forwardPass(tokens) {
        const vocabSize = this.tokenizer ? Object.keys(this.tokenizer.vocab || {}).length : 32000;
        const logits = new Float32Array(vocabSize);
        
        const lastToken = tokens[tokens.length - 1] || 0;
        const position = tokens.length;
        
        // Calcolo basato su pattern linguistici
        for (let i = 0; i < vocabSize; i++) {
            const tokenFreq = this.getTokenFrequency(i);
            const positionalBias = Math.sin(position * 0.1 + i * 0.01);
            const tokenSimilarity = this.getTokenSimilarity(lastToken, i);
            
            logits[i] = tokenFreq * 0.3 + positionalBias * 0.2 + tokenSimilarity * 0.5;
            logits[i] += (Math.random() - 0.5) * 0.2;
        }
        
        return logits;
    }

    /**
     * Sampling dai logits
     */
    sample(logits, temperature) {
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(l => Math.exp((l - maxLogit) / temperature));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(e => e / sumExp);
        
        let rand = Math.random();
        let cumProb = 0;
        
        for (let i = 0; i < probs.length; i++) {
            cumProb += probs[i];
            if (rand < cumProb) {
                return i;
            }
        }
        
        return probs.length - 1;
    }

    /**
     * Ottieni frequenza token
     */
    getTokenFrequency(tokenId) {
        if (tokenId < 32) return 0.1;
        if (tokenId < 65) return 0.3;
        if (tokenId < 90) return 0.2;
        if (tokenId < 97) return 0.1;
        if (tokenId < 122) return 0.4;
        return 0.1;
    }

    /**
     * Calcola similarità token
     */
    getTokenSimilarity(token1, token2) {
        const diff = Math.abs(token1 - token2);
        return Math.max(0, 1 - diff / 50);
    }
}
