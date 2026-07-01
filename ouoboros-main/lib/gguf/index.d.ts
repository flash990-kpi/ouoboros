/**
 * Ouroboros GGUF Parser - TypeScript Definitions
 * Production-ready type definitions for GGUF file streaming
 */

export enum GGUFValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

export type MetadataBaseValue = string | number | bigint | boolean;
export type MetadataValue = MetadataBaseValue | MetadataBaseValue[] | MetadataValue[];

export type Version = 1 | 2 | 3;

export interface GGUFTensorInfo {
  name: string;
  n_dims: number;
  shape: bigint[];
  dtype: number;
  offset: bigint;
}

export interface GGUFMetadata {
  version: Version;
  tensor_count: bigint;
  kv_count: bigint;
  [key: string]: MetadataValue;
}

export interface GGUFTypedMetadataEntry {
  value: MetadataValue;
  type: GGUFValueType;
  subType?: GGUFValueType;
}

export interface GGUFTypedMetadata {
  version: {
    value: Version;
    type: GGUFValueType.UINT32;
  };
  tensor_count: {
    value: bigint;
    type: GGUFValueType.UINT32 | GGUFValueType.UINT64;
  };
  kv_count: {
    value: bigint;
    type: GGUFValueType.UINT32 | GGUFValueType.UINT64;
  };
  [key: string]: GGUFTypedMetadataEntry;
}

export interface GGUFParseOutput {
  metadata: GGUFMetadata;
  tensorInfos: GGUFTensorInfo[];
  tensorDataOffset: bigint;
  littleEndian: boolean;
  tensorInfoByteRange: [number, number];
  parameterCount?: number;
  typedMetadata?: GGUFTypedMetadata;
}

export interface GGUFParseOptions {
  fetch?: typeof fetch;
  additionalFetchHeaders?: Record<string, string>;
  typedMetadata?: boolean;
  computeParametersCount?: boolean;
  allowLocalFile?: boolean;
}

/**
 * Parse a GGUF file (streaming mode - no full file load)
 * Optimized for browser and Node.js environments
 * 
 * @param uri - File URI (HTTP/HTTPS or local path)
 * @param params - Parse options
 * @returns Promise resolving to parsed GGUF structure
 */
export function gguf(uri: string, params?: GGUFParseOptions): Promise<GGUFParseOutput>;

export const GGUF_MAGIC_NUMBER: Uint8Array;