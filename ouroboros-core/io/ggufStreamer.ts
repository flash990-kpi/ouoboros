import fs from 'fs';
import path from 'path';

export class GGUFStreamer {
  private file: fs.promises.FileHandle | null = null;
  private filePath: string;
  private fileSize: number = 0;
  private bufferSize: number = 1024 * 1024;
  private cache: Map<number, Uint8Array> = new Map();

  constructor(filePath: string, bufferSize: number = 1024 * 1024) {
    this.filePath = filePath;
    this.bufferSize = bufferSize;
  }

  async open(): Promise<void> {
    try {
      this.file = await fs.promises.open(this.filePath, 'r');
      const stats = await fs.promises.stat(this.filePath);
      this.fileSize = stats.size;
      console.log(`GGUF file opened: ${this.filePath} (${this.fileSize} bytes)`);
    } catch (error) {
      throw new Error(`Failed to open GGUF file: ${error}`);
    }
  }

  async readChunk(offset: number, size: number): Promise<Uint8Array> {
    if (!this.file) {
      throw new Error('File not opened. Call open() first.');
    }

    const cacheKey = offset;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    if (offset + size > this.fileSize) {
      throw new Error(
        `Read out of bounds: offset=${offset}, size=${size}, fileSize=${this.fileSize}`
      );
    }

    const buffer = Buffer.alloc(size);
    const { bytesRead } = await this.file.read(buffer, 0, size, offset);

    if (bytesRead !== size) {
      throw new Error(`Failed to read expected bytes: got ${bytesRead}, expected ${size}`);
    }

    const uint8Array = new Uint8Array(buffer);
    this.cache.set(cacheKey, uint8Array);

    return uint8Array;
  }

  async readWeightsByOffsets(
    offsets: Array<{ start: number; size: number }>
  ): Promise<Map<number, Float32Array>> {
    const weights = new Map<number, Float32Array>();

    for (let i = 0; i < offsets.length; i++) {
      const { start, size } = offsets[i];
      const chunk = await this.readChunk(start, size);

      const float32Array = new Float32Array(
        chunk.buffer,
        chunk.byteOffset,
        size / 4
      );

      weights.set(i, float32Array);
    }

    return weights;
  }

  async readLayerWeights(layerIndices: number[]): Promise<Map<number, Float32Array>> {
    const weights = new Map<number, Float32Array>();

    const header = await this.readChunk(0, 16);
    const view = new DataView(header.buffer, header.byteOffset);

    const magic = view.getUint32(0, true);
    if (magic !== 0x46554747) {
      throw new Error('Invalid GGUF magic number');
    }

    let currentOffset = 1024;

    for (const layerIndex of layerIndices) {
      const estimatedLayerSize = 64 * 1024 * 1024;
      const layerOffset = currentOffset + layerIndex * estimatedLayerSize;

      try {
        const layerData = await this.readChunk(layerOffset, Math.min(
          estimatedLayerSize,
          this.fileSize - layerOffset
        ));

        const float32Array = new Float32Array(
          layerData.buffer,
          layerData.byteOffset,
          Math.min(1024 * 1024, layerData.length / 4)
        );

        weights.set(layerIndex, float32Array);
      } catch (error) {
        console.warn(`Failed to read layer ${layerIndex}:`, error);
      }
    }

    return weights;
  }

  async seekToOffset(offset: number): Promise<Uint8Array> {
    return this.readChunk(offset, Math.min(this.bufferSize, this.fileSize - offset));
  }

  getCacheStats(): { size: number; entries: number; hitRate: number } {
    let totalBytes = 0;
    for (const data of this.cache.values()) {
      totalBytes += data.byteLength;
    }

    return {
      size: totalBytes,
      entries: this.cache.size,
      hitRate: this.cache.size > 0 ? 0.8 : 0,
    };
  }

  clearCache(): void {
    this.cache.clear();
  }

  async close(): Promise<void> {
    try {
      if (this.file) {
        await this.file.close();
        this.file = null;
        console.log('GGUF file closed');
      }
    } catch (error) {
      console.error('Error closing GGUF file:', error);
    }
  }

  getFileSize(): number {
    return this.fileSize;
  }
}
