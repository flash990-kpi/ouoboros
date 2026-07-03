export class TopologyParser {
    parseIndex(buffer) {
        const view = new DataView(buffer);
        const magic = view.getUint32(0, true);
        if (magic !== TopologyParser.MAGIC_OURO) {
            throw new Error("Firma magica del file .ouro corrotta o non identificabile.");
        }
        const version = view.getUint32(4, true);
        const tensorCount = view.getUint32(8, true);
        const layerCount = view.getUint32(12, true);
        const topology = {
            version,
            tensorCount,
            layerCount,
            records: new Map(),
            layerGroups: new Map()
        };
        const headerOffset = 32;
        const recordSize = 48;
        for (let i = 0; i < tensorCount; i++) {
            const offset = headerOffset + (i * recordSize);
            if (offset + recordSize > buffer.byteLength)
                break;
            // Read 64-bit values as two 32-bit parts
            const tensorHashLow = view.getUint32(offset, true);
            const tensorHashHigh = view.getUint32(offset + 4, true);
            const tensorHash = tensorHashLow + (tensorHashHigh * 0x100000000);
            
            const offsetLow = view.getUint32(offset + 8, true);
            const offsetHigh = view.getUint32(offset + 12, true);
            const ggufOffset = offsetLow + (offsetHigh * 0x100000000);
            
            const byteLengthLow = view.getUint32(offset + 16, true);
            const byteLengthHigh = view.getUint32(offset + 20, true);
            const byteLength = byteLengthLow + (byteLengthHigh * 0x100000000);
            
            const layerIndex = view.getUint32(offset + 24, true);
            const tensorType = view.getUint32(offset + 28, true);
            const sparsityRank = view.getUint32(offset + 32, true);
            const record = {
                tensorHash,
                ggufOffset,
                byteLength,
                layerIndex,
                tensorType,
                sparsityRank,
                shape: [1, byteLength / 4] // Placeholder shape: [1, elements]
            };
            topology.records.set(tensorHash, record);
            if (!topology.layerGroups.has(layerIndex)) {
                topology.layerGroups.set(layerIndex, []);
            }
            topology.layerGroups.get(layerIndex).push(record);
        }
        return topology;
    }
    static hashTensorName(name) {
        // FNV-1a hash using Number instead of BigInt for browser compatibility
        const encoder = new TextEncoder();
        const data = encoder.encode(name);
        let hash = 2166136261; // FNV offset basis (32-bit)
        const prime = 16777619; // FNV prime (32-bit)
        for (let i = 0; i < data.length; i++) {
            hash = hash ^ data[i];
            hash = (hash * prime) >>> 0; // Force unsigned 32-bit
        }
        return hash;
    }
}
TopologyParser.MAGIC_OURO = 0x4f55524f; // 'OURO'
