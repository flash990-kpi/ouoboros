export interface OuroTensorRecord {
    tensorHash: bigint;
    ggufOffset: bigint;
    byteLength: bigint;
    layerIndex: number;
    tensorType: number;
    sparsityRank: number;
}

export interface OuroTopologyMap {
    version: number;
    tensorCount: number;
    layerCount: number;
    records: Map<bigint, OuroTensorRecord>;
    layerGroups: Map<number, OuroTensorRecord[]>;
}

export class TopologyParser {
    private static readonly MAGIC_OURO = 0x4f55524f; // 'OURO'

    public parseIndex(buffer: ArrayBuffer): OuroTopologyMap {
        const view = new DataView(buffer);
        const magic = view.getUint32(0, true);
        if (magic !== TopologyParser.MAGIC_OURO) {
            throw new Error("Firma magica del file .ouro corrotta o non identificabile.");
        }

        const version = view.getUint32(4, true);
        const tensorCount = view.getUint32(8, true);
        const layerCount = view.getUint32(12, true);

        const topology: OuroTopologyMap = {
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
            if (offset + recordSize > buffer.byteLength) break;

            const tensorHash = view.getBigUint64(offset, true);
            const ggufOffset = view.getBigUint64(offset + 8, true);
            const byteLength = view.getBigUint64(offset + 16, true);
            const layerIndex = view.getUint32(offset + 24, true);
            const tensorType = view.getUint32(offset + 28, true);
            const sparsityRank = view.getUint32(offset + 32, true);

            const record: OuroTensorRecord = {
                tensorHash,
                ggufOffset,
                byteLength,
                layerIndex,
                tensorType,
                sparsityRank
            };

            topology.records.set(tensorHash, record);
            if (!topology.layerGroups.has(layerIndex)) {
                topology.layerGroups.set(layerIndex, []);
            }
            topology.layerGroups.get(layerIndex)!.push(record);
        }

        return topology;
    }

    public static hashTensorName(name: string): bigint {
        const encoder = new TextEncoder();
        const data = encoder.encode(name);
        let hash = 14695981039346656037n;
        const prime = 1099511628211n;

        for (let i = 0; i < data.length; i++) {
            hash = hash ^ BigInt(data[i]);
            hash = (hash * prime) & 0xFFFFFFFFFFFFFFFFn;
        }
        return hash;
    }
}

