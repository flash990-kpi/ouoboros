import { HardwareAuditor } from '../hw/auditor.js';
import { WebGpuDriver } from '../hw/webgpu_driver.js';
import { WebNnDriver } from '../hw/webnn_driver.js';
import { WasmDriver } from '../hw/wasm_driver.js';
import { TopologyParser } from '../asts/topologyParser.js';
import { SparsityPredictor } from '../asts/sparsityPredictor.js';
import { WeightSynthesizer } from '../asts/weightSynthesizer.js';
import { GgufStreamer } from '../io/ggufStreamer.js';
import { Scheduler } from './scheduler.js';
import { LLMDecoder } from './llmDecoder.js';
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
        // GGUF file e LLM decoder per inferenza vera
        this.localGgufFile = null;
        this.ggufMetadata = null;
        this.llmDecoder = null;
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
            
            // Estrai metadata GGUF per A.S.T.S.
            const { gguf } = await import('https://esm.sh/@huggingface/gguf@0.4.2');
            const fileUrl = URL.createObjectURL(source.fileObject);
            const { metadata } = await gguf(fileUrl);
            this.ggufMetadata = metadata;
            
            // Inizializza LLM decoder per inferenza vera
            this.llmDecoder = new LLMDecoder(this.ggufMetadata);
            this.llmDecoder.extractVocabulary();
            
            console.log('[STATE MACHINE] A.S.T.S. system ready with LLM decoder');
            console.log('[STATE MACHINE] GGUF Metadata:', metadata);
            
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
     * Usa LLM decoder per vera inferenza con file GGUF locale.
     */
    submitInference(prompt, onTokenGenerated) {
        if (this.internalState === 'BOOTSTRAPPING' || this.internalState === 'ERROR') {
            throw new Error(`Invocazione di inferenza non consentita nello stato corrente: ${this.internalState}`);
        }
        
        if (!this.llmDecoder) {
            throw new Error('LLM decoder non inizializzato');
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
                    
                    // VERA INFERENZA LLM con decoder
                    this.transitionTo('EXECUTION', { layer: 0 });
                    
                    // Generazione con streaming dei token usando LLM decoder
                    const generator = this.llmDecoder.generate(prompt, 200, 0.7);
                    
                    for await (const token of generator) {
                        onTokenGenerated(token, {
                            layer: 0,
                            rank: routingPath.targetRank,
                            compression: routingPath.dynamicCompressionRatio,
                            activeNodes: routingPath.requiredTensors.length
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
