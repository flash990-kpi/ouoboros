import { HardwareAuditor } from '../hw/auditor.js';
import { WebGpuDriver } from '../hw/webgpu_driver.js';
import { WebNnDriver } from '../hw/webnn_driver.js';
import { WasmDriver } from '../hw/wasm_driver.js';
import { TopologyParser } from '../asts/topologyParser.js';
import { SparsityPredictor } from '../asts/sparsityPredictor.js';
import { WeightSynthesizer } from '../asts/weightSynthesizer.js';
import { GgufStreamer } from '../io/ggufStreamer.js';
import { Scheduler } from './scheduler.js';
import { pipeline, env } from '@xenova/transformers';

// Configura Transformers.js per usare file locali
env.allowLocalModels = true;
env.useBrowserCache = false;
export class OuroborosKernel {
    constructor(stateChangeNotifier) {
        this.internalState = 'BOOTSTRAPPING';
        this.executionScheduler = new Scheduler();
        // Driver Core
        this.driverWebGpu = new WebGpuDriver();
        this.driverWebNn = new WebNnDriver();
        this.driverWasm = new WasmDriver();
        // Moduli ASTS e I/O
        this.topologyParser = new TopologyParser();
        this.weightSynthesizer = new WeightSynthesizer();
        this.onStateChange = stateChangeNotifier;
        // Transformers.js pipeline per vera inferenza LLM
        this.transformersPipeline = null;
    }
    /**
     * Inizializza l'architettura. Se il buffer .ouro è null,
     * innesca automaticamente l'estrazione e generazione dall'header GGUF.
     */
    async boot(source, ouroBuffer) {
        try {
            this.transitionTo('BOOTSTRAPPING');
            const auditor = new HardwareAuditor();
            this.hardwareProfile = await auditor.profileDevice();
            
            // Forza WASM driver per supporto completo quantizzazione GGUF
            // WebGPU non supporta dequantizzazione Q4_K nativamente
            console.log('[STATE MACHINE] Forcing WASM driver for GGUF quantization support');
            this.hardwareProfile.primaryDriver = 'WASM';
            
            // Inizializzazione del driver concordato dall'auditor hardware
            if (this.hardwareProfile.primaryDriver === 'WebNN') {
                await this.driverWebNn.initialize();
            }
            else if (this.hardwareProfile.primaryDriver === 'WebGPU') {
                await this.driverWebGpu.initialize();
            }
            else {
                await this.driverWasm.initialize();
            }
            this.fileStreamer = new GgufStreamer(source);
            let finalOuroBuffer;
            if (!ouroBuffer) {
                // Generazione dinamica a caldo dall'header del GGUF se non è pre-indicizzato
                finalOuroBuffer = await this.fileStreamer.generateTopologyFromGguf();
            }
            else {
                finalOuroBuffer = ouroBuffer;
            }
            this.activeTopologyMap = this.topologyParser.parseIndex(finalOuroBuffer);
            this.sparsityPredictor = new SparsityPredictor(this.activeTopologyMap);
            
            // Inizializza Transformers.js con il file GGUF locale
            console.log('[STATE MACHINE] Initializing Transformers.js pipeline...');
            const fileUrl = URL.createObjectURL(source.fileObject);
            this.transformersPipeline = await pipeline('text-generation', fileUrl, {
                dtype: 'q4',
                device: 'webgpu',
                progress_callback: (progress) => {
                    console.log('[TRANSFORMERS] Progress:', progress);
                }
            });
            console.log('[STATE MACHINE] Transformers.js pipeline initialized');
            
            this.transitionTo('IDLE', {
                driver: this.hardwareProfile.primaryDriver,
                tensors: this.activeTopologyMap.tensorCount,
                layers: this.activeTopologyMap.layerCount
            });
        }
        catch (error) {
            console.error('[STATE MACHINE] Boot error:', error);
            this.transitionTo('ERROR', { reason: error.message });
        }
    }
    /**
     * Inietta un prompt nel motore di navigazione geometrica A.S.T.S.
     * Usa Transformers.js per vera inferenza LLM.
     */
    submitInference(prompt, onTokenGenerated) {
        console.log('[STATE MACHINE] submitInference called with prompt:', prompt);
        console.log('[STATE MACHINE] Current state:', this.internalState);
        
        if (this.internalState === 'BOOTSTRAPPING' || this.internalState === 'ERROR') {
            throw new Error(`Invocazione di inferenza non consentita nello stato corrente: ${this.internalState}`);
        }
        
        if (!this.transformersPipeline) {
            throw new Error('Transformers.js pipeline non inizializzato');
        }
        
        console.log('[STATE MACHINE] Enqueuing inference task...');
        this.executionScheduler.enqueue({
            id: `inference_session_${performance.now()}`,
            priority: 100,
            action: async () => {
                try {
                    console.log('[STATE MACHINE] Starting real LLM inference with Transformers.js...');
                    this.transitionTo('ANALYSIS');
                    
                    // FASE DI ANALISI GEOMETRICA A.S.T.S.
                    console.log('[STATE MACHINE] Predicting routing path...');
                    const routingPath = this.sparsityPredictor.predictRoutingPath(prompt);
                    console.log('[STATE MACHINE] Routing path:', routingPath);
                    
                    this.transitionTo('SYNTHESIS', {
                        hash: 0,
                        index: 0,
                        total: routingPath.requiredTensors.length
                    });
                    
                    // VERA INFERENZA LLM con Transformers.js
                    console.log('[STATE MACHINE] Running Transformers.js text generation...');
                    const output = await this.transformersPipeline(prompt, {
                        max_new_tokens: 100,
                        do_sample: true,
                        temperature: 0.7,
                        top_k: 50,
                        top_p: 0.95,
                        callback_function: (outputItem) => {
                            // Callback per streaming token
                            if (outputItem.token && outputItem.token.text !== null) {
                                const token = outputItem.token.text;
                                console.log('[STATE MACHINE] Generated token:', token);
                                onTokenGenerated(token, {
                                    layer: 0,
                                    rank: routingPath.targetRank,
                                    compression: routingPath.dynamicCompressionRatio,
                                    activeNodes: routingPath.requiredTensors.length
                                });
                            }
                        }
                    });
                    
                    this.transitionTo('EXECUTION', { layer: 0 });
                    console.log('[STATE MACHINE] Transformers.js output:', output);
                    
                    this.transitionTo('IDLE');
                    console.log('[STATE MACHINE] Real LLM inference completed successfully');
                }
                catch (err) {
                    console.error('[STATE MACHINE] Inference error:', err);
                    this.transitionTo('ERROR', { reason: err.message });
                }
            }
        });
    }
    transitionTo(newState, payload) {
        this.internalState = newState;
        this.onStateChange(newState, payload);
    }
    get currentEngineState() {
        return this.internalState;
    }
}
