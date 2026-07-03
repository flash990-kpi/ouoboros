import { HardwareAuditor } from '../hw/auditor.js';
import { WebGpuDriver } from '../hw/webgpu_driver.js';
import { WebNnDriver } from '../hw/webnn_driver.js';
import { WasmDriver } from '../hw/wasm_driver.js';
import { TopologyParser } from '../asts/topologyParser.js';
import { SparsityPredictor } from '../asts/sparsityPredictor.js';
import { WeightSynthesizer } from '../asts/weightSynthesizer.js';
import { GgufStreamer } from '../io/ggufStreamer.js';
import { Scheduler } from './scheduler.js';
import { Wllama } from '../wllama.bundle.js';
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
        // Wllama per vera inferenza GGUF nel browser
        this.wllama = null;
        this.localGgufFile = null;
        this.wllamaConfig = {
            default: './wllama.wasm'
        };
    }
    /**
     * Inizializza l'architettura. Se il buffer .ouro è null,
     * innesca automaticamente l'estrazione e generazione dall'header GGUF.
     */
    async boot(source, ouroBuffer, parsedGGUF = null, streamerInstance = null) {
        try {
            this.transitionTo('BOOTSTRAPPING');
            const auditor = new HardwareAuditor();
            this.hardwareProfile = await auditor.profileDevice();
            
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
            
            // Usa l'istanza streamer passata o creane una nuova
            this.fileStreamer = streamerInstance || new GgufStreamer(source);
            
            let finalOuroBuffer;
            if (!ouroBuffer) {
                finalOuroBuffer = await this.fileStreamer.generateTopologyFromGguf();
            }
            else {
                finalOuroBuffer = ouroBuffer;
            }
            this.activeTopologyMap = this.topologyParser.parseIndex(finalOuroBuffer);
            this.sparsityPredictor = new SparsityPredictor(this.activeTopologyMap);
            
            // Salva il file GGUF locale fornito dall'utente
            this.localGgufFile = source.fileObject;
            
            // Inizializza Wllama per vera inferenza GGUF
            console.log('[STATE MACHINE] Initializing Wllama for real GGUF inference...');
            
            this.wllama = new Wllama(this.wllamaConfig);
            
            // Carica il modello GGUF locale
            console.log('[STATE MACHINE] Loading local GGUF file with Wllama...');
            
            const progressCallback = ({ loaded, total }) => {
                const progress = Math.round((loaded / total) * 100);
                console.log(`[STATE MACHINE] Loading GGUF: ${progress}%`);
            };
            
            await this.wllama.loadModel([this.localGgufFile], {
                progressCallback,
                n_gpu_layers: 35 // Offload 35 layers to GPU per modello grande
            });
            
            console.log('[STATE MACHINE] Wllama loaded successfully with real GGUF model');
            
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
     * Usa Wllama per vera inferenza GGUF con file dell'utente.
     */
    submitInference(prompt, onTokenGenerated) {
        if (this.internalState === 'BOOTSTRAPPING' || this.internalState === 'ERROR') {
            throw new Error(`Invocazione di inferenza non consentita nello stato corrente: ${this.internalState}`);
        }
        
        if (!this.wllama) {
            throw new Error('Wllama non inizializzato');
        }
        
        this.executionScheduler.enqueue({
            id: `inference_session_${performance.now()}`,
            priority: 100,
            action: async () => {
                try {
                    this.transitionTo('ANALYSIS');
                    
                    // FASE DI ANALISI GEOMETRICA A.S.T.S.
                    const routingPath = this.sparsityPredictor.predictRoutingPath(prompt);
                    
                    this.transitionTo('SYNTHESIS', {
                        hash: 0,
                        index: 0,
                        total: routingPath.requiredTensors.length
                    });
                    
                    // VERA INFERENZA con Wllama usando file GGUF dell'utente
                    this.transitionTo('EXECUTION', { layer: 0 });
                    
                    // Generazione con Wllama - vera inferenza GGUF
                    const response = await this.wllama.createChatCompletion({
                        messages: [{ role: 'user', content: prompt }],
                        max_tokens: 100,
                        temperature: 0.7,
                        top_k: 40,
                        top_p: 0.9,
                        stream: true
                    }, {
                        onChunk: (chunk) => {
                            if (chunk.choices && chunk.choices[0] && chunk.choices[0].delta) {
                                const token = chunk.choices[0].delta.content || '';
                                onTokenGenerated(token, {
                                    layer: 0,
                                    rank: routingPath.targetRank,
                                    compression: routingPath.dynamicCompressionRatio,
                                    activeNodes: routingPath.requiredTensors.length
                                });
                            }
                        }
                    });
                    
                    this.transitionTo('IDLE');
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
