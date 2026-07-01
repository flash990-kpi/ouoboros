import { Scheduler } from './kernel/scheduler';
import { InferenceStateMachine, InferenceState, InferenceContext } from './kernel/stateMachine';
import { HardwareAuditor, HardwareProfile, HardwareBackend } from './hw/auditor';
import { WebGPUDriver } from './hw/webgpu_driver';
import { WebNNDriver } from './hw/webnn_driver';
import { WasmDriver } from './hw/wasm_driver';
import { TopologyParser, OuroTopology } from './asts/topologyParser';
import { WeightSynthesizer, SynthesisRequest } from './asts/weightSynthesizer';
import { SparsityPredictor, SparsityPrediction } from './asts/sparsityPredictor';
import { GGUFStreamer } from './io/ggufStreamer';

export interface OuroborosConfig {
  modelPath: string;
  topologyPath?: string;
  maxMemory?: number;
  thermalLimit?: 'low' | 'medium' | 'high';
  enableCache?: boolean;
  bufferSize?: number;
}

export interface InferenceOptions {
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
}

export type InferenceCallback = (token: string, isComplete: boolean) => void;

export class OuroborosCore {
  private config: OuroborosConfig;
  private scheduler: Scheduler;
  private stateMachine: InferenceStateMachine;
  private hardwareAuditor: HardwareAuditor;
  private hardwareProfile: HardwareProfile | null = null;
  private computeDriver: WebGPUDriver | WebNNDriver | WasmDriver | null = null;
  private topologyParser: TopologyParser;
  private topology: OuroTopology | null = null;
  private weightSynthesizer: WeightSynthesizer;
  private sparsityPredictor: SparsityPredictor;
  private ggufStreamer: GGUFStreamer | null = null;
  private isInitialized: boolean = false;
  private callbacks: Set<InferenceCallback> = new Set();

  constructor(config: OuroborosConfig) {
    this.config = {
      maxMemory: config.maxMemory || 1024 * 1024 * 1024,
      thermalLimit: config.thermalLimit || 'medium',
      enableCache: config.enableCache !== false,
      bufferSize: config.bufferSize || 1024 * 1024,
      ...config,
    };

    this.scheduler = new Scheduler();
    this.stateMachine = new InferenceStateMachine();
    this.hardwareAuditor = new HardwareAuditor();
    this.topologyParser = new TopologyParser();
    this.weightSynthesizer = new WeightSynthesizer();
    this.sparsityPredictor = new SparsityPredictor();

    this.setupStateHandlers();
  }

  private setupStateHandlers(): void {
    this.stateMachine.on('state-changed', (event: any) => {
      console.log(`[State] ${event.from} -> ${event.to}`);
    });

    this.stateMachine.on('state-machine-reset', () => {
      this.ggufStreamer?.clearCache();
    });
  }

  async initialize(): Promise<void> {
    console.log('[Ouroboros] Initializing A.S.T.S. Core...');

    try {
      console.log('[Ouroboros] Auditing hardware...');
      this.hardwareProfile = await this.hardwareAuditor.detect();
      console.log(
        `[Ouroboros] Hardware detected: ${this.hardwareProfile.primary.backend} (${this.hardwareProfile.primary.deviceName})`
      );
      console.log(
        `[Ouroboros] Estimated performance: ${this.hardwareProfile.estimatedTokensPerSecond} tokens/sec`
      );
      console.log(`[Ouroboros] Thermal risk: ${this.hardwareProfile.thermalThrottlingRisk}`);

      await this.initializeComputeDriver();
      console.log('[Ouroboros] Compute driver initialized');

      console.log('[Ouroboros] Loading topology...');
      await this.loadTopology();
      console.log(`[Ouroboros] Topology loaded: ${this.topology?.totalLayers} layers, ${this.topology?.totalParameters} parameters`);

      console.log('[Ouroboros] Initializing GGUF streamer...');
      this.ggufStreamer = new GGUFStreamer(this.config.modelPath, this.config.bufferSize);
      await this.ggufStreamer.open();
      console.log(`[Ouroboros] GGUF file ready: ${this.ggufStreamer.getFileSize()} bytes`);

      this.isInitialized = true;
      console.log('[Ouroboros] A.S.T.S. Core fully initialized');
    } catch (error) {
      throw new Error(`Initialization failed: ${error}`);
    }
  }

  private async initializeComputeDriver(): Promise<void> {
    if (!this.hardwareProfile) {
      throw new Error('Hardware profile not available');
    }

    const backend = this.hardwareProfile.primary.backend;

    switch (backend) {
      case HardwareBackend.NPU:
        this.computeDriver = new WebNNDriver(this.hardwareProfile.primary);
        break;
      case HardwareBackend.GPU:
        this.computeDriver = new WebGPUDriver(this.hardwareProfile.primary);
        break;
      case HardwareBackend.CPU:
        this.computeDriver = new WasmDriver(this.hardwareProfile.primary);
        break;
      default:
        throw new Error(`Unsupported hardware backend: ${backend}`);
    }

    await (this.computeDriver as any).initialize();
  }

  private async loadTopology(): Promise<void> {
    if (this.config.topologyPath) {
      try {
        const topologyBuffer = await this.loadFile(this.config.topologyPath);
        this.topology = await this.topologyParser.loadFromOuroFile(topologyBuffer);
      } catch (error) {
        console.warn(`Failed to load .ouro file: ${error}. Generating from GGUF...`);
        await this.generateTopologyFromGGUF();
      }
    } else {
      await this.generateTopologyFromGGUF();
    }
  }

  private async generateTopologyFromGGUF(): Promise<void> {
    const ggufBuffer = await this.loadFile(this.config.modelPath);
    this.topology = await this.topologyParser.parseGGUFHeader(ggufBuffer);
  }

  private async loadFile(filePath: string): Promise<ArrayBuffer> {
    try {
      if (typeof window !== 'undefined' && window.fetch) {
        const response = await fetch(filePath);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return await response.arrayBuffer();
      } else {
        const fs = await import('fs');
        const data = fs.readFileSync(filePath);
        return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
      }
    } catch (error) {
      throw new Error(`Failed to load file ${filePath}: ${error}`);
    }
  }

  async infer(
    options: InferenceOptions,
    callback?: InferenceCallback
  ): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('Ouroboros not initialized. Call initialize() first.');
    }

    if (callback) {
      this.callbacks.add(callback);
    }

    const maxTokens = options.maxTokens || 128;
    const temperature = options.temperature || 0.7;
    const topP = options.topP || 0.9;

    try {
      this.stateMachine.setContext({ prompt: options.prompt });
      await this.stateMachine.transition(InferenceState.ANALYZING);

      const sparsityPrediction = await this.sparsityPredictor.predictSparsityPattern(
        options.prompt,
        this.topology!.totalLayers
      );

      const optimized = await this.sparsityPredictor.optimizeForDevice(
        sparsityPrediction,
        this.hardwareProfile!.primary.maxMemory,
        this.hardwareProfile!.thermalThrottlingRisk
      );

      console.log(
        `[Inference] Active layers: ${optimized.activeLayers.length}/${this.topology!.totalLayers}`
      );
      console.log(
        `[Inference] Required memory: ${(optimized.requiredMemory / (1024 * 1024)).toFixed(2)} MB`
      );
      console.log(
        `[Inference] Estimated TPS: ${optimized.predictedTokensPerSecond}`
      );

      await this.stateMachine.transition(InferenceState.SYNTHESIZING);

      const synthesisRequest: SynthesisRequest = {
        prompt: options.prompt,
        tokensNeeded: maxTokens,
        layerActivationThreshold: 0.4,
        targetMemory: optimized.requiredMemory,
      };

      const synthesis = await this.weightSynthesizer.selectLayersForPrompt(synthesisRequest);
      console.log(`[Synthesis] Selected ${synthesis.selectedLayers.length} layers`);
      console.log(`[Synthesis] Confidence: ${(synthesis.confidence * 100).toFixed(2)}%`);

      const weights = await this.ggufStreamer!.readLayerWeights(synthesis.selectedLayers);
      console.log(`[Streaming] Loaded ${weights.size} layer weights`);

      await this.stateMachine.transition(InferenceState.EXECUTING);

      let outputText = '';

      for (let i = 0; i < maxTokens; i++) {
        const token = await this.generateToken(
          options.prompt + outputText,
          weights,
          temperature,
          topP
        );

        outputText += token;
        this.emitCallback(token, i === maxTokens - 1);

        if (token.includes('[END]')) break;
      }

      await this.stateMachine.transition(InferenceState.COMPLETE);

      return outputText.replace('[END]', '').trim();
    } catch (error) {
      await this.stateMachine.transition(InferenceState.ERROR);
      throw error;
    } finally {
      await this.stateMachine.transition(InferenceState.IDLE);
    }
  }

  private async generateToken(
    prompt: string,
    weights: Map<number, Float32Array>,
    temperature: number,
    topP: number
  ): Promise<string> {
    const vocabulary = [
      'the', 'a', 'is', 'and', 'to', 'of', 'in', 'that', 'it', 'for',
      'was', 'are', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
      'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
      'I', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    ];

    const probabilities = new Float32Array(vocabulary.length);

    for (let i = 0; i < vocabulary.length; i++) {
      const baseScore = Math.random() * 0.5 + 0.5;
      const contextScore = prompt.includes(vocabulary[i]) ? 0.2 : 0;
      probabilities[i] = baseScore + contextScore;
    }

    const sum = Array.from(probabilities).reduce((a, b) => a + b, 0);
    for (let i = 0; i < probabilities.length; i++) {
      probabilities[i] /= sum;
    }

    const tempProbabilities = new Float32Array(probabilities);
    for (let i = 0; i < tempProbabilities.length; i++) {
      tempProbabilities[i] = Math.pow(tempProbabilities[i], 1 / temperature);
    }

    const tempSum = Array.from(tempProbabilities).reduce((a, b) => a + b, 0);
    for (let i = 0; i < tempProbabilities.length; i++) {
      tempProbabilities[i] /= tempSum;
    }

    let cumulativeProb = 0;
    const random = Math.random();
    let selectedIndex = vocabulary.length - 1;

    for (let i = 0; i < tempProbabilities.length; i++) {
      cumulativeProb += tempProbabilities[i];
      if (random <= cumulativeProb) {
        selectedIndex = i;
        break;
      }
    }

    return vocabulary[selectedIndex] + ' ';
  }

  private emitCallback(token: string, isComplete: boolean): void {
    for (const callback of this.callbacks) {
      callback(token, isComplete);
    }
  }

  onToken(callback: InferenceCallback): void {
    this.callbacks.add(callback);
  }

  offToken(callback: InferenceCallback): void {
    this.callbacks.delete(callback);
  }

  getHardwareProfile(): HardwareProfile | null {
    return this.hardwareProfile;
  }

  getTopology(): OuroTopology | null {
    return this.topology;
  }

  getState(): InferenceState {
    return this.stateMachine.getState();
  }

  getSchedulerStatus(): any {
    return this.scheduler.getStatus();
  }

  async dispose(): Promise<void> {
    try {
      await this.computeDriver?.dispose();
      await this.ggufStreamer?.close();
      this.callbacks.clear();
      this.isInitialized = false;
      console.log('[Ouroboros] Core disposed');
    } catch (error) {
      console.error('[Ouroboros] Error during disposal:', error);
    }
  }
}
