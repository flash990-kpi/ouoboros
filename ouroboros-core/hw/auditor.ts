export enum HardwareBackend {
  NPU = 'NPU',
  GPU = 'GPU',
  CPU = 'CPU',
}

export interface HardwareCapabilities {
  backend: HardwareBackend;
  maxMemory: number;
  computeUnits: number;
  maxWorkgroupSize: number;
  supportsFloat32: boolean;
  supportsFloat16: boolean;
  supportsBFloat16: boolean;
  maxBufferSize: number;
  deviceName: string;
}

export interface HardwareProfile {
  primary: HardwareCapabilities;
  fallbacks: HardwareCapabilities[];
  thermalThrottlingRisk: 'low' | 'medium' | 'high';
  estimatedTokensPerSecond: number;
}

export class HardwareAuditor {
  private profile: HardwareProfile | null = null;
  private detectionTime: number = 0;

  async detect(): Promise<HardwareProfile> {
    const startTime = performance.now();
    const backends: HardwareCapabilities[] = [];

    try {
      const npuCaps = await this.detectNPU();
      if (npuCaps) backends.push(npuCaps);
    } catch (e) {
      console.debug('NPU detection skipped');
    }

    try {
      const gpuCaps = await this.detectGPU();
      if (gpuCaps) backends.push(gpuCaps);
    } catch (e) {
      console.debug('GPU detection skipped');
    }

    try {
      const cpuCaps = await this.detectCPU();
      if (cpuCaps) backends.push(cpuCaps);
    } catch (e) {
      console.debug('CPU detection skipped');
    }

    if (backends.length === 0) {
      throw new Error('No suitable hardware acceleration found');
    }

    backends.sort((a, b) => this.scoreBackend(b) - this.scoreBackend(a));

    const profile: HardwareProfile = {
      primary: backends[0],
      fallbacks: backends.slice(1),
      thermalThrottlingRisk: this.estimateThermalRisk(backends[0]),
      estimatedTokensPerSecond: this.estimatePerformance(backends[0]),
    };

    this.profile = profile;
    this.detectionTime = performance.now() - startTime;
    return profile;
  }

  private async detectNPU(): Promise<HardwareCapabilities | null> {
    if (!('ml' in navigator)) {
      return null;
    }

    try {
      const context = (navigator as any).ml.createContextSync();
      return {
        backend: HardwareBackend.NPU,
        maxMemory: 512 * 1024 * 1024,
        computeUnits: 8,
        maxWorkgroupSize: 256,
        supportsFloat32: true,
        supportsFloat16: true,
        supportsBFloat16: true,
        maxBufferSize: 256 * 1024 * 1024,
        deviceName: 'Neural Processing Unit (WebNN)',
      };
    } catch (e) {
      return null;
    }
  }

  private async detectGPU(): Promise<HardwareCapabilities | null> {
    if (!navigator.gpu) {
      return null;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return null;

      const limits = adapter.limits;
      const device = await adapter.requestDevice();

      return {
        backend: HardwareBackend.GPU,
        maxMemory: device.limits.maxBufferSize,
        computeUnits: limits.maxComputeWorkgroupsPerDimension || 65535,
        maxWorkgroupSize: limits.maxComputeInvocationsPerWorkgroup || 256,
        supportsFloat32: true,
        supportsFloat16: true,
        supportsBFloat16: false,
        maxBufferSize: limits.maxBufferSize || 4 * 1024 * 1024 * 1024,
        deviceName: adapter.name || 'WebGPU Device',
      };
    } catch (e) {
      return null;
    }
  }

  private async detectCPU(): Promise<HardwareCapabilities | null> {
    try {
      if (!WebAssembly.simd) {
        throw new Error('SIMD not supported');
      }

      return {
        backend: HardwareBackend.CPU,
        maxMemory: (navigator.deviceMemory || 4) * 1024 * 1024 * 1024,
        computeUnits: navigator.hardwareConcurrency || 4,
        maxWorkgroupSize: navigator.hardwareConcurrency || 4,
        supportsFloat32: true,
        supportsFloat16: false,
        supportsBFloat16: false,
        maxBufferSize: 2 * 1024 * 1024 * 1024,
        deviceName: `CPU (${navigator.hardwareConcurrency || 1} cores, WASM+SIMD)`,
      };
    } catch (e) {
      return null;
    }
  }

  private scoreBackend(caps: HardwareCapabilities): number {
    const baseScores: Record<HardwareBackend, number> = {
      [HardwareBackend.NPU]: 1000,
      [HardwareBackend.GPU]: 800,
      [HardwareBackend.CPU]: 100,
    };

    let score = baseScores[caps.backend];
    score += (caps.computeUnits / 32) * 100;
    score += (caps.maxMemory / (4 * 1024 * 1024 * 1024)) * 50;

    if (caps.supportsFloat16) score += 50;
    if (caps.supportsBFloat16) score += 30;

    return score;
  }

  private estimateThermalRisk(
    caps: HardwareCapabilities
  ): 'low' | 'medium' | 'high' {
    if (caps.backend === HardwareBackend.NPU) return 'low';
    if (caps.backend === HardwareBackend.GPU) {
      if (caps.computeUnits > 16) return 'medium';
      return 'low';
    }
    return 'high';
  }

  private estimatePerformance(caps: HardwareCapabilities): number {
    const basePerf: Record<HardwareBackend, number> = {
      [HardwareBackend.NPU]: 128,
      [HardwareBackend.GPU]: 64,
      [HardwareBackend.CPU]: 8,
    };

    let perf = basePerf[caps.backend];
    perf *= caps.computeUnits / 4;

    return Math.round(perf);
  }

  getProfile(): HardwareProfile | null {
    return this.profile;
  }

  getDetectionTime(): number {
    return this.detectionTime;
  }
}
