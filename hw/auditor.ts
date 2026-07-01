export interface HardwareProfile {
    primaryDriver: 'WebNN' | 'WebGPU' | 'WASM_SIMD';
    maxBufferSize: number;
    maxComputeWorkgroupStorageSize: number;
    threadsAvailable: number;
    npuSupported: boolean;
    profilingTimeMs: number;
}

export class HardwareAuditor {
    private static readonly WASM_SIMD_MAGIC = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x60, 0x00, 0x00, 0x03, 0x02,
        0x01, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0xfd,
        0x0b, 0x01, 0x00, 0x0b
    ]);

    public async profileDevice(): Promise<HardwareProfile> {
        const startTime = performance.now();
        const profile: HardwareProfile = {
            primaryDriver: 'WASM_SIMD',
            maxBufferSize: 2147483648, 
            maxComputeWorkgroupStorageSize: 32768,
            threadsAvailable: navigator.hardwareConcurrency || 4,
            npuSupported: false,
            profilingTimeMs: 0
        };

        // 1. Rilevamento Accelerazione NPU (WebNN)
        if (typeof navigator !== 'undefined' && 'ml' in navigator) {
            try {
                // @ts-ignore
                const context = await navigator.ml.createContext({ deviceType: 'npu' });
                if (context) {
                    profile.npuSupported = true;
                    profile.primaryDriver = 'WebNN';
                    profile.profilingTimeMs = performance.now() - startTime;
                    return profile;
                }
            } catch (e) {
                profile.npuSupported = false;
            }
        }

        // 2. Rilevamento Accelerazione GPU (WebGPU)
        if ('gpu' in navigator) {
            try {
                const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
                if (adapter) {
                    profile.primaryDriver = 'WebGPU';
                    profile.maxBufferSize = adapter.limits.maxBufferSize;
                    profile.maxComputeWorkgroupStorageSize = adapter.limits.maxComputeWorkgroupStorageSize;
                    profile.profilingTimeMs = performance.now() - startTime;
                    return profile;
                }
            } catch (e) {}
        }

        // 3. Controllo Fallback CPU SIMD 128-bit
        const hasSimd = WebAssembly.validate(HardwareAuditor.WASM_SIMD_MAGIC);
        if (!hasSimd) {
            throw new Error("L'architettura hardware in uso non supporta le istruzioni parallele WASM SIMD a 128-bit.");
        }

        profile.profilingTimeMs = performance.now() - startTime;
        return profile;
    }
}
