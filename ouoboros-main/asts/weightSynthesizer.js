export class WeightSynthesizer {
    /**
     * Esegue l'un-packing bit-a-bit e la scomposizione matematica dei tensori quantizzati,
     * allocando il risultato all'interno di un buffer condiviso ad alta velocità.
     */
    synthesizeTensor(rawBuffer, type, rank) {
        const totalBytes = rawBuffer.byteLength;
        // Calcolo della dimensione finale stimata in base al tipo
        let outputElements;
        if (type === 0) {
            // Q4_0: 2 float32 per byte
            outputElements = totalBytes * 2;
        } else if (type === 14) {
            // I8: 1 float32 per byte
            outputElements = totalBytes;
        } else {
            // Float32: 1 float32 per 4 byte
            outputElements = totalBytes / 4;
        }
        const sharedBuffer = new SharedArrayBuffer(outputElements * 4);
        const outputView = new Float32Array(sharedBuffer);
        const inputView = new DataView(rawBuffer);
        // Coefficiente di riscalatura della topologia sparsa
        const scaleFactor = 2.0 / (1.0 + Math.exp(-rank / 2.0));
        
        if (type === 0) {
            // Algoritmo di Dequantizzazione Q4_0 puro ad alte prestazioni (32 elementi per blocco)
            // Struttura blocco GGUF Q4_0: 2 byte f16 (scale) + 16 byte pesi (32 nibble)
            const blockSize = 18;
            const totalBlocks = Math.floor(totalBytes / blockSize);
            for (let b = 0; b < totalBlocks; b++) {
                const blockOffset = b * blockSize;
                // Lettura dello scale del blocco a 16-bit float (convertito manualmente in f32)
                const rawScaleF16 = inputView.getUint16(blockOffset, true);
                const sign = (rawScaleF16 & 0x8000) >> 15;
                const exponent = (rawScaleF16 & 0x7C00) >> 10;
                const mantissa = rawScaleF16 & 0x03FF;
                let scaleF32 = 0;
                if (exponent === 0x1F) {
                    scaleF32 = mantissa !== 0 ? NaN : (sign !== 0 ? -Infinity : Infinity);
                }
                else if (exponent === 0) {
                    scaleF32 = (sign !== 0 ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024.0);
                }
                else {
                    scaleF32 = (sign !== 0 ? -1 : 1) * Math.pow(2, exponent - 15) * (1.0 + mantissa / 1024.0);
                }
                const finalScale = scaleF32 * scaleFactor;
                // Estrazione sequenziale dei 32 nibbles dai successivi 16 byte
                for (let i = 0; i < 16; i++) {
                    const byteVal = inputView.getUint8(blockOffset + 2 + i);
                    const lowNibble = byteVal & 0x0F;
                    const highNibble = (byteVal >> 4) & 0x0F;
                    const baseOutputIndex = (b * 32) + (i * 2);
                    if (baseOutputIndex + 1 < outputView.length) {
                        outputView[baseOutputIndex] = (lowNibble - 8) * finalScale;
                        outputView[baseOutputIndex + 1] = (highNibble - 8) * finalScale;
                    }
                }
            }
        }
        else if (type === 14) {
            // Dequantizzazione I8 (int8) per tensori di input/embedding
            const elements = Math.floor(totalBytes);
            for (let i = 0; i < elements && i < outputView.length; i++) {
                const int8Val = inputView.getInt8(i);
                outputView[i] = int8Val * scaleFactor;
            }
        }
        else {
            // Dequantizzazione e riscalatura per tensori lineari Float32 nativi
            const elements = Math.floor(totalBytes / 4);
            for (let i = 0; i < elements; i++) {
                outputView[i] = inputView.getFloat32(i * 4, true) * scaleFactor;
            }
        }
        return sharedBuffer;
    }
}
