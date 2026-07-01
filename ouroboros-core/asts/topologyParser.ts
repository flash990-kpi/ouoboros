export interface OuroTopology {
  version: string;
  modelName: string;
  totalLayers: number;
  totalParameters: number;
  layerDimensions: number[][];
  clusterMap: Map<number, Cluster>;
  rankInformation: RankInfo[];
  compressionRatio: number;
}

export interface Cluster {
  id: number;
  layerIndices: number[];
  startOffset: number;
  endOffset: number;
  size: number;
  rank: number;
  frequency: number;
}

export interface RankInfo {
  layer: number;
  rank: number;
  compressionFactor: number;
  reconstructionError: number;
}

export class TopologyParser {
  private topology: OuroTopology | null = null;
  private cache: Map<number, Uint8Array> = new Map();

  async parseGGUFHeader(ggufBuffer: ArrayBuffer): Promise<OuroTopology> {
    const view = new DataView(ggufBuffer);
    const magic = view.getUint32(0, true);

    if (magic !== 0x46554747) {
      throw new Error('Invalid GGUF file: magic number mismatch');
    }

    const version = view.getUint32(4, true);
    const tensorCount = view.getUint32(8, true);
    const metadataKVCount = view.getUint32(12, true);

    const topology: OuroTopology = {
      version: `GGUF v${version}`,
      modelName: 'Unknown',
      totalLayers: 0,
      totalParameters: 0,
      layerDimensions: [],
      clusterMap: new Map(),
      rankInformation: [],
      compressionRatio: 1.0,
    };

    let offset = 16;

    for (let i = 0; i < metadataKVCount; i++) {
      const kvPair = this.parseMetadataKV(view, offset);
      offset += kvPair.bytesRead;

      if (kvPair.key === 'general.name') {
        topology.modelName = kvPair.value as string;
      }
    }

    const layers: Array<{ name: string; shape: number[]; size: number }> = [];
    let totalParams = 0;

    for (let i = 0; i < tensorCount; i++) {
      const tensorInfo = this.parseTensorInfo(view, offset);
      offset += tensorInfo.bytesRead;

      if (tensorInfo.name.includes('weight') || tensorInfo.name.includes('attn')) {
        layers.push({
          name: tensorInfo.name,
          shape: tensorInfo.shape,
          size: tensorInfo.size,
        });
        totalParams += tensorInfo.size / 4;
      }
    }

    topology.totalLayers = layers.length;
    topology.totalParameters = totalParams;
    topology.layerDimensions = layers.map((l) => l.shape);

    this.generateClusters(topology, layers);
    this.generateRankInformation(topology);

    this.topology = topology;
    return topology;
  }

  private parseMetadataKV(
    view: DataView,
    offset: number
  ): { key: string; value: unknown; bytesRead: number } {
    let pos = offset;
    const keyLen = view.getUint32(pos, true);
    pos += 4;

    const keyBytes = new Uint8Array(
      view.buffer,
      view.byteOffset + pos,
      keyLen
    );
    const key = new TextDecoder().decode(keyBytes);
    pos += keyLen;

    const valueType = view.getUint32(pos, true);
    pos += 4;

    let value: unknown = null;
    let valueSize = 0;

    switch (valueType) {
      case 0: // uint8
        value = view.getUint8(pos);
        valueSize = 1;
        break;
      case 1: // int8
        value = view.getInt8(pos);
        valueSize = 1;
        break;
      case 4: // uint32
        value = view.getUint32(pos, true);
        valueSize = 4;
        break;
      case 6: // uint64
        value = view.getBigUint64(pos, true);
        valueSize = 8;
        break;
      case 20: // string
        const strLen = view.getUint32(pos, true);
        pos += 4;
        const strBytes = new Uint8Array(
          view.buffer,
          view.byteOffset + pos,
          strLen
        );
        value = new TextDecoder().decode(strBytes);
        valueSize = 4 + strLen;
        break;
    }

    return {
      key,
      value,
      bytesRead: keyLen + 4 + 4 + valueSize,
    };
  }

  private parseTensorInfo(
    view: DataView,
    offset: number
  ): { name: string; shape: number[]; size: number; bytesRead: number } {
    let pos = offset;

    const nameLen = view.getUint32(pos, true);
    pos += 4;
    const nameBytes = new Uint8Array(
      view.buffer,
      view.byteOffset + pos,
      nameLen
    );
    const name = new TextDecoder().decode(nameBytes);
    pos += nameLen;

    const ndim = view.getUint32(pos, true);
    pos += 4;
    const shape: number[] = [];
    for (let i = 0; i < ndim; i++) {
      shape.push(Number(view.getBigUint64(pos, true)));
      pos += 8;
    }

    const type = view.getUint32(pos, true);
    pos += 4;

    const offset_tensor = view.getBigUint64(pos, true);
    pos += 8;

    let elementSize = 4;
    switch (type) {
      case 0:
        elementSize = 4;
        break;
      case 1:
        elementSize = 2;
        break;
      case 2:
        elementSize = 1;
        break;
    }

    const size = shape.reduce((a, b) => a * b, 1) * elementSize;

    return {
      name,
      shape,
      size,
      bytesRead: 4 + nameLen + 4 + ndim * 8 + 4 + 8,
    };
  }

  private generateClusters(
    topology: OuroTopology,
    layers: Array<{ name: string; shape: number[]; size: number }>
  ): void {
    const clusterSize = Math.ceil(layers.length / 8);
    let startOffset = 0;

    for (let i = 0; i < layers.length; i += clusterSize) {
      const clusterLayers = layers.slice(i, Math.min(i + clusterSize, layers.length));
      const size = clusterLayers.reduce((sum, l) => sum + l.size, 0);

      const cluster: Cluster = {
        id: Math.floor(i / clusterSize),
        layerIndices: Array.from({ length: clusterLayers.length }, (_, idx) => i + idx),
        startOffset,
        endOffset: startOffset + size,
        size,
        rank: Math.max(1, Math.floor(Math.sqrt(size / (4 * 1024)))),
        frequency: 1,
      };

      topology.clusterMap.set(cluster.id, cluster);
      startOffset += size;
    }
  }

  private generateRankInformation(topology: OuroTopology): void {
    for (let i = 0; i < topology.totalLayers; i++) {
      const dims = topology.layerDimensions[i] || [1];
      const maxDim = Math.max(...dims);
      const minDim = Math.min(...dims);
      const rank = Math.max(1, Math.floor((minDim * 0.7) / 8));
      const compressionFactor = maxDim / Math.max(1, rank);

      topology.rankInformation.push({
        layer: i,
        rank,
        compressionFactor,
        reconstructionError: 0.02 * Math.log(compressionFactor),
      });
    }

    const totalError = topology.rankInformation.reduce((sum, r) => sum + r.reconstructionError, 0);
    topology.compressionRatio = topology.totalParameters / Math.max(1, totalError);
  }

  async loadFromOuroFile(buffer: ArrayBuffer): Promise<OuroTopology> {
    const view = new DataView(buffer);
    const magic = view.getUint32(0, true);

    if (magic !== 0x4f555250) {
      throw new Error('Invalid .ouro file: magic number mismatch (0x4f555250)');
    }

    const version = view.getUint32(4, true);
    const modelNameLen = view.getUint32(8, true);
    let offset = 12;

    const modelNameBytes = new Uint8Array(
      buffer,
      offset,
      modelNameLen
    );
    const modelName = new TextDecoder().decode(modelNameBytes);
    offset += modelNameLen;

    const totalLayers = view.getUint32(offset, true);
    offset += 4;
    const totalParameters = view.getBigUint64(offset, true);
    offset += 8;

    const layerDimensions: number[][] = [];
    for (let i = 0; i < totalLayers; i++) {
      const ndim = view.getUint32(offset, true);
      offset += 4;
      const dims: number[] = [];
      for (let j = 0; j < ndim; j++) {
        dims.push(Number(view.getBigUint64(offset, true)));
        offset += 8;
      }
      layerDimensions.push(dims);
    }

    const clusterCount = view.getUint32(offset, true);
    offset += 4;
    const clusterMap = new Map<number, Cluster>();

    for (let i = 0; i < clusterCount; i++) {
      const clusterId = view.getUint32(offset, true);
      offset += 4;
      const layerCount = view.getUint32(offset, true);
      offset += 4;
      const layerIndices: number[] = [];
      for (let j = 0; j < layerCount; j++) {
        layerIndices.push(view.getUint32(offset, true));
        offset += 4;
      }
      const startOffset = Number(view.getBigUint64(offset, true));
      offset += 8;
      const endOffset = Number(view.getBigUint64(offset, true));
      offset += 8;
      const rank = view.getUint32(offset, true);
      offset += 4;
      const frequency = view.getUint32(offset, true);
      offset += 4;

      clusterMap.set(clusterId, {
        id: clusterId,
        layerIndices,
        startOffset,
        endOffset,
        size: endOffset - startOffset,
        rank,
        frequency,
      });
    }

    const rankCount = view.getUint32(offset, true);
    offset += 4;
    const rankInformation: RankInfo[] = [];

    for (let i = 0; i < rankCount; i++) {
      const layer = view.getUint32(offset, true);
      offset += 4;
      const rank = view.getUint32(offset, true);
      offset += 4;
      const compressionFactor = view.getFloat32(offset, true);
      offset += 4;
      const reconstructionError = view.getFloat32(offset, true);
      offset += 4;

      rankInformation.push({
        layer,
        rank,
        compressionFactor,
        reconstructionError,
      });
    }

    const compressionRatio = view.getFloat32(offset, true);

    this.topology = {
      version: `Ouro v${version}`,
      modelName,
      totalLayers,
      totalParameters: Number(totalParameters),
      layerDimensions,
      clusterMap,
      rankInformation,
      compressionRatio,
    };

    return this.topology;
  }

  getTopology(): OuroTopology | null {
    return this.topology;
  }

  getCluster(clusterId: number): Cluster | undefined {
    return this.topology?.clusterMap.get(clusterId);
  }

  getRankInfo(layerIndex: number): RankInfo | undefined {
    return this.topology?.rankInformation[layerIndex];
  }
}
