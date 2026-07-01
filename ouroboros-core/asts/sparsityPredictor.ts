export interface SparsityPrediction {
  activeLayers: number[];
  sparsityPattern: Uint8Array;
  estimatedComplexity: number;
  predictedTokensPerSecond: number;
  requiredMemory: number;
}

export interface LayerActivation {
  layerIndex: number;
  activationScore: number;
  isActive: boolean;
  estimatedFlops: number;
}

export class SparsityPredictor {
  private activationCache: Map<string, LayerActivation[]> = new Map();
  private patternCache: Map<string, Uint8Array> = new Map();

  async predictSparsityPattern(prompt: string, totalLayers: number): Promise<SparsityPrediction> {
    const cacheKey = `${prompt.slice(0, 100)}-${totalLayers}`;

    if (this.activationCache.has(cacheKey)) {
      return this.generatePredictionFromCache(cacheKey, totalLayers);
    }

    const promptLength = prompt.length;
    const wordCount = prompt.split(/\s+/).length;
    const uniqueTokens = new Set(prompt.split(/\s+/)).size;

    const layerActivations: LayerActivation[] = [];

    for (let i = 0; i < totalLayers; i++) {
      const baseScore = Math.sin((i * Math.PI) / totalLayers) + 0.5;
      const complexityBonus = (uniqueTokens / wordCount) * 0.3;
      const depthPenalty = Math.exp(-i / (totalLayers / 4)) * 0.2;

      const activationScore = Math.min(
        1.0,
        baseScore + complexityBonus + depthPenalty
      );

      const threshold = 0.35 + (i / totalLayers) * 0.2;

      layerActivations.push({
        layerIndex: i,
        activationScore,
        isActive: activationScore > threshold,
        estimatedFlops: activationScore * (1024 * 1024),
      });
    }

    this.activationCache.set(cacheKey, layerActivations);

    const activeLayers = layerActivations
      .filter((la) => la.isActive)
      .map((la) => la.layerIndex);

    const sparsityPattern = new Uint8Array(totalLayers);
    for (let i = 0; i < totalLayers; i++) {
      sparsityPattern[i] = layerActivations[i].isActive ? 1 : 0;
    }

    this.patternCache.set(cacheKey, sparsityPattern);

    const totalFlops = layerActivations.reduce((sum, la) => sum + la.estimatedFlops, 0);
    const estimatedComplexity = Math.log2(totalFlops) / Math.log2(1024 * 1024);

    const memoryPerLayer = 64 * 1024 * 1024;
    const requiredMemory = activeLayers.length * memoryPerLayer;

    return {
      activeLayers,
      sparsityPattern,
      estimatedComplexity,
      predictedTokensPerSecond: Math.max(
        8,
        Math.min(256, Math.floor(2048 / Math.exp(estimatedComplexity / 10)))
      ),
      requiredMemory,
    };
  }

  private generatePredictionFromCache(
    cacheKey: string,
    totalLayers: number
  ): SparsityPrediction {
    const activations = this.activationCache.get(cacheKey)!;
    const pattern = this.patternCache.get(cacheKey)!;

    const activeLayers = activations
      .filter((a) => a.isActive)
      .map((a) => a.layerIndex);

    const totalFlops = activations.reduce((sum, a) => sum + a.estimatedFlops, 0);
    const estimatedComplexity = Math.log2(totalFlops) / Math.log2(1024 * 1024);

    return {
      activeLayers,
      sparsityPattern: pattern,
      estimatedComplexity,
      predictedTokensPerSecond: Math.max(
        8,
        Math.min(256, Math.floor(2048 / Math.exp(estimatedComplexity / 10)))
      ),
      requiredMemory: activeLayers.length * 64 * 1024 * 1024,
    };
  }

  async optimizeForDevice(
    prediction: SparsityPrediction,
    deviceMemory: number,
    thermalLimit: 'low' | 'medium' | 'high'
  ): Promise<SparsityPrediction> {
    let activeLayers = [...prediction.activeLayers];

    const thermalFactors: Record<string, number> = {
      low: 0.6,
      medium: 0.8,
      high: 1.0,
    };

    const maxLayers = Math.floor(
      (deviceMemory * thermalFactors[thermalLimit]) / (64 * 1024 * 1024)
    );

    if (activeLayers.length > maxLayers) {
      activeLayers = activeLayers.slice(0, maxLayers);
    }

    const optimizedPattern = new Uint8Array(prediction.sparsityPattern.length);
    for (const layer of activeLayers) {
      optimizedPattern[layer] = 1;
    }

    const requiredMemory = activeLayers.length * 64 * 1024 * 1024;
    const tpsReduction = Math.min(1.0, requiredMemory / prediction.requiredMemory);

    return {
      activeLayers,
      sparsityPattern: optimizedPattern,
      estimatedComplexity: prediction.estimatedComplexity,
      predictedTokensPerSecond: Math.floor(
        prediction.predictedTokensPerSecond * tpsReduction
      ),
      requiredMemory,
    };
  }

  getActivationScore(layerIndex: number, cacheKey: string): number | null {
    const activations = this.activationCache.get(cacheKey);
    if (!activations) return null;

    return activations[layerIndex]?.activationScore ?? null;
  }
}
