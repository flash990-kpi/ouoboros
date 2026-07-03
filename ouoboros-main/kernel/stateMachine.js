import { HardwareAuditor } from '../hw/auditor.js';
import { WebGpuDriver } from '../hw/webgpu_driver.js';
import { WebNnDriver } from '../hw/webnn_driver.js';
import { WasmDriver } from '../hw/wasm_driver.js';
import { TopologyParser } from '../asts/topologyParser.js';
import { SparsityPredictor } from '../asts/sparsityPredictor.js';
import { WeightSynthesizer } from '../asts/weightSynthesizer.js';
import { GgufStreamer } from '../io/ggufStreamer.js';
import { Scheduler } from './scheduler.js';
import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.0/esm/index.min.js';
import { HfInference } from 'https://cdn.jsdelivr.net/npm/@huggingface/inference@2.6.4/dist/index.min.js';
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
        // Wllama per vera inferenza GGUF locale (modelli piccoli)
        this.wllama = null;
        this.localGgufFile = null;
        this.wllamaConfig = {
            default: 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.1.0/esm/wasm/wllama.wasm'
        };
        
        // Hugging Face Inference API per cloud (modelli grandi/mobile)
        this.hfInference = null;
        this.hfApiKey = null; // Da configurare
        this.useCloudInference = false;
        
        // Sistema ibrido intelligente
        this.inferenceMode = 'local'; // 'local', 'cloud', 'hybrid'
        this.deviceType = 'desktop'; // 'desktop', 'mobile'
    }
    
    /**
     * Detect device type (desktop vs mobile)
     */
    detectDeviceType() {
        const userAgent = navigator.userAgent;
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent);
        this.deviceType = isMobile ? 'mobile' : 'desktop';
        console.log('[STATE MACHINE] Device type detected:', this.deviceType);
        return this.deviceType;
    }
    
    /**
     * Determine optimal inference mode based on device and model size
     */
    determineInferenceMode(modelSize) {
        this.detectDeviceType();
        
        const MODEL_SIZE_THRESHOLD = 4 * 1024 * 1024 * 1024; // 4GB
        
        if (this.deviceType === 'mobile') {
            // Mobile: sempre usa cloud Hugging Face
            this.inferenceMode = 'cloud';
            console.log('[STATE MACHINE] Mobile device: using cloud inference (Hugging Face)');
        } else if (modelSize > MODEL_SIZE_THRESHOLD) {
            // Desktop + modello grande: cloud o server locale
            this.inferenceMode = 'cloud';
            console.log('[STATE MACHINE] Large model (>4GB): using cloud inference');
        } else {
            // Desktop + modello piccolo: locale wllama
            this.inferenceMode = 'local';
            console.log('[STATE MACHINE] Small model (<4GB): using local inference (wllama)');
        }
        
        return this.inferenceMode;
    }
    
    /**
     * Set Hugging Face API key for cloud inference
     */
    setHfApiKey(apiKey) {
        this.hfApiKey = apiKey;
        if (this.hfInference) {
            this.hfInference = new HfInference(apiKey);
        }
        console.log('[STATE MACHINE] Hugging Face API key set');
    }
    
    /**
     * Set model ID for cloud inference
     */
    setCloudModelId(modelId) {
        this.cloudModelId = modelId;
        console.log('[STATE MACHINE] Cloud model ID set:', modelId);
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
            
            // Determina modalità inferenza ottimale
            const modelSize = this.localGgufFile ? this.localGgufFile.size : 0;
            this.determineInferenceMode(modelSize);
            
            if (this.inferenceMode === 'cloud') {
                // Cloud inference con Hugging Face
                console.log('[STATE MACHINE] Initializing Hugging Face Inference API...');
                
                // Usa API key se configurata, altrimenti inference gratuita
                this.hfApiKey = this.hfApiKey || null;
                this.hfInference = new HfInference(this.hfApiKey);
                
                console.log('[STATE MACHINE] Cloud inference ready (no local model loading needed)');
            } else {
                // Local inference con wllama
                console.log('[STATE MACHINE] Initializing Wllama for local GGUF inference...');
                
                this.wllama = new Wllama(this.wllamaConfig);
                
                // Carica il modello GGUF locale
                console.log('[STATE MACHINE] Loading local GGUF file with Wllama...');
                console.log('[STATE MACHINE] File size:', this.localGgufFile.size, 'bytes (', (this.localGgufFile.size / 1e9).toFixed(2), 'GB)');
                
                const progressCallback = ({ loaded, total }) => {
                    const progress = Math.round((loaded / total) * 100);
                    console.log(`[STATE MACHINE] Loading GGUF: ${progress}%`);
                };
                
                try {
                    await this.wllama.loadModel([this.localGgufFile], {
                        progressCallback,
                        n_gpu_layers: 35, // Offload 35 layers to GPU per modello grande
                        n_ctx: 2048, // Context size
                        use_mmap: true, // Use memory mapping for large files
                    });
                    
                    console.log('[STATE MACHINE] Wllama loaded successfully with real GGUF model');
                } catch (error) {
                    console.error('[STATE MACHINE] Wllama load error:', error);
                    if (error.message.includes('offset is out of bounds') || error.message.includes('memory')) {
                        // Fallback to cloud inference if local fails
                        console.warn('[STATE MACHINE] Local inference failed, falling back to cloud...');
                        this.inferenceMode = 'cloud';
                        this.hfInference = new HfInference(this.hfApiKey);
                    } else {
                        throw error;
                    }
                }
            }
            
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
     * Usa sistema ibrido: locale (wllama) o cloud (Hugging Face) in base al dispositivo/modello.
     */
    submitInference(prompt, onTokenGenerated) {
        if (this.internalState === 'BOOTSTRAPPING' || this.internalState === 'ERROR') {
            throw new Error(`Invocazione di inferenza non consentita nello stato corrente: ${this.internalState}`);
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
                    
                    this.transitionTo('EXECUTION', { layer: 0 });
                    
                    if (this.inferenceMode === 'cloud') {
                        // CLOUD INFERENCE con Hugging Face
                        console.log('[STATE MACHINE] Using cloud inference (Hugging Face)');
                        
                        // Usa modello Hugging Face appropriato (può essere configurato)
                        const modelId = this.cloudModelId || 'meta-llama/Llama-3.1-70B-Instruct';
                        
                        const stream = await this.hfInference.textGeneration({
                            model: modelId,
                            inputs: prompt,
                            parameters: {
                                max_new_tokens: 100,
                                temperature: 0.7,
                                top_k: 40,
                                top_p: 0.9,
                                return_full_text: false
                            }
                        });
                        
                        // Streaming della risposta
                        for await (const chunk of stream) {
                            const token = chunk.token?.text || '';
                            if (token) {
                                onTokenGenerated(token, {
                                    layer: 0,
                                    rank: routingPath.targetRank,
                                    compression: routingPath.dynamicCompressionRatio,
                                    activeNodes: routingPath.requiredTensors.length,
                                    inferenceMode: 'cloud'
                                });
                            }
                        }
                    } else {
                        // LOCAL INFERENCE con wllama
                        console.log('[STATE MACHINE] Using local inference (wllama)');
                        
                        if (!this.wllama) {
                            throw new Error('Wllama non inizializzato');
                        }
                        
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
                                        activeNodes: routingPath.requiredTensors.length,
                                        inferenceMode: 'local'
                                    });
                                }
                            }
                        });
                    }
                    
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
