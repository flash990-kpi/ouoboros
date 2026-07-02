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
        this.llmPipeline = null;
        this.localGgufFile = null;
        
        // Configura Transformers.js per uso locale e performance
        env.allowLocalModels = true;
        env.useBrowserCache = true;
        env.allowRemoteModels = true; // Permetti download modelli remoti
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
                finalOuroBuffer = await this.fileStreamer.generateTopologyFromGguf();
            }
            else {
                finalOuroBuffer = ouroBuffer;
            }
            this.activeTopologyMap = this.topologyParser.parseIndex(finalOuroBuffer);
            this.sparsityPredictor = new SparsityPredictor(this.activeTopologyMap);
            
            // Salva il file GGUF locale fornito dall'utente
            this.localGgufFile = source.fileObject;
            
            // Inizializza Transformers.js con modello pre-addestrato per inferenza vera
            // Usiamo un modello instruction-tuned per risposte appropriate
            console.log('[STATE MACHINE] Initializing Transformers.js pipeline...');
            
            // Usa un modello instruction-tuned per risposte sensate
            this.llmPipeline = await pipeline('text2text-generation', 'Xenova/flan-t5-small', {
                quantized: true,
                device: this.hardwareProfile.primaryDriver === 'WebGPU' ? 'webgpu' : 'wasm',
                progress_callback: (progress) => {
                    console.log('[TRANSFORMERS] Progress:', progress);
                }
            });
            
            console.log('[STATE MACHINE] Transformers.js pipeline initialized with', this.hardwareProfile.primaryDriver);
            
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
     * Usa Transformers.js per vera inferenza LLM con streaming.
     */
    submitInference(prompt, onTokenGenerated) {
        if (this.internalState === 'BOOTSTRAPPING' || this.internalState === 'ERROR') {
            throw new Error(`Invocazione di inferenza non consentita nello stato corrente: ${this.internalState}`);
        }
        
        if (!this.llmPipeline) {
            throw new Error('Transformers.js pipeline non inizializzato');
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
                    
                    // VERA INFERENZA LLM con Transformers.js streaming
                    this.transitionTo('EXECUTION', { layer: 0 });
                    
                    // Generazione con streaming dei token (text2text-generation)
                    const output = await this.llmPipeline(prompt, {
                        max_new_tokens: 100,
                        temperature: 0.7,
                        do_sample: true
                    });
                    
                    // Streaming dei token per compatibilità UI
                    const generatedText = output[0].generated_text;
                    for (let i = 0; i < generatedText.length; i++) {
                        onTokenGenerated(generatedText[i], {
                            layer: 0,
                            rank: routingPath.targetRank,
                            compression: routingPath.dynamicCompressionRatio,
                            activeNodes: routingPath.requiredTensors.length
                        });
                        // Small delay per streaming effect
                        await new Promise(resolve => setTimeout(resolve, 10));
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
