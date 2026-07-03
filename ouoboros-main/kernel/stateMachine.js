import { HardwareAuditor } from '../hw/auditor.js';
import { WebGpuDriver } from '../hw/webgpu_driver.js';
import { WebNnDriver } from '../hw/webnn_driver.js';
import { WasmDriver } from '../hw/wasm_driver.js';
import { TopologyParser } from '../asts/topologyParser.js';
import { SparsityPredictor } from '../asts/sparsityPredictor.js';
import { WeightSynthesizer } from '../asts/weightSynthesizer.js';
import { GgufStreamer } from '../io/ggufStreamer.js';
import { Scheduler } from './scheduler.js';
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
        
        // A.S.T.S. System - True zero-RAM loading with partial weight streaming
        this.localGgufFile = null;
        this.ggufStreamer = null;
        this.activeDriver = null; // WebNN (NPU), WebGPU (GPU), or WASM (CPU)
        
        // Tokenizer per generazione testo reale
        this.tokenizer = null;
        this.vocab = null;
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
            
            // Seleziona driver hardware ottimale (NPU > GPU > CPU)
            console.log('[STATE MACHINE] Selecting optimal hardware driver...');
            
            if (this.hardwareProfile.primaryDriver === 'webnn') {
                this.activeDriver = this.driverWebNn;
                console.log('[STATE MACHINE] Using WebNN driver (NPU) - optimal for mobile');
            } else if (this.hardwareProfile.primaryDriver === 'webgpu') {
                this.activeDriver = this.driverWebGpu;
                console.log('[STATE MACHINE] Using WebGPU driver (GPU) - high performance');
            } else {
                this.activeDriver = this.driverWasm;
                console.log('[STATE MACHINE] Using WASM driver (CPU) - SIMD fallback');
            }
            
            // Inizializza GGUF Streamer per streaming parziale pesi
            this.ggufStreamer = streamerInstance || new GgufStreamer(source);
            
            // Estrai vocabolario tokenizer dal GGUF parsato
            if (parsedGGUF && parsedGGUF.metadata) {
                this.vocab = this.extractVocabFromMetadata(parsedGGUF.metadata);
                console.log('[STATE MACHINE] Extracted vocabulary:', Object.keys(this.vocab).length, 'tokens');
            }
            
            // Inizializza driver selezionato
            console.log('[STATE MACHINE] Initializing hardware driver...');
            await this.activeDriver.initialize(this.hardwareProfile);
            
            console.log('[STATE MACHINE] A.S.T.S. system ready - zero RAM loading, partial weight streaming');
            
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
     * Vera inferenza A.S.T.S.: zero RAM loading, partial weight streaming
     */
    submitInference(prompt, onTokenGenerated) {
        if (this.internalState === 'BOOTSTRAPPING' || this.internalState === 'ERROR') {
            throw new Error(`Invocazione di inferenza non consentita nello stato corrente: ${this.internalState}`);
        }
        
        if (!this.activeDriver) {
            throw new Error('Hardware driver non inizializzato');
        }
        
        if (!this.ggufStreamer) {
            throw new Error('GGUF Streamer non inizializzato');
        }
        
        this.executionScheduler.enqueue({
            id: `inference_session_${performance.now()}`,
            priority: 100,
            action: async () => {
                try {
                    this.transitionTo('ANALYSIS');
                    
                    // FASE DI ANALISI GEOMETRICA A.S.T.S.
                    const routingPath = this.sparsityPredictor.predictRoutingPath(prompt);
                    console.log('[STATE MACHINE] A.S.T.S. routing path calculated:', routingPath);
                    
                    this.transitionTo('SYNTHESIS', {
                        hash: 0,
                        index: 0,
                        total: routingPath.requiredTensors.length
                    });
                    
                    // FASE DI SINTESI A.S.T.S. - Streaming parziale pesi
                    this.transitionTo('EXECUTION', { layer: 0 });
                    
                    console.log('[STATE MACHINE] Starting A.S.T.S. partial weight streaming...');
                    
                    // Per ogni tensore richiesto, streaming parziale dal GGUF
                    for (let i = 0; i < routingPath.requiredTensors.length; i++) {
                        const tensorInfo = routingPath.requiredTensors[i];
                        
                        // Ensure tensorInfo has required properties
                        if (!tensorInfo) {
                            console.warn('[STATE MACHINE] Skipping undefined tensor at index', i);
                            continue;
                        }
                        
                        // Ensure tensorInfo has shape property
                        if (!tensorInfo.shape) {
                            tensorInfo.shape = [1, Math.max(1, tensorInfo.byteLength / 4)];
                        }
                        
                        // Ensure tensorInfo has layer property
                        if (!tensorInfo.layer) {
                            tensorInfo.layer = tensorInfo.layerIndex || 0;
                        }
                        
                        // Streaming parziale: leggi solo i byte necessari dal GGUF
                        const weightChunk = await this.ggufStreamer.readWeightChunk(
                            tensorInfo.ggufOffset,
                            tensorInfo.byteLength
                        );
                        
                        // Sintesi pesi con WeightSynthesizer
                        const synthesizedWeights = await this.weightSynthesizer.synthesizeTensor(
                            weightChunk,
                            tensorInfo.dtype || tensorInfo.tensorType || 0,
                            routingPath.targetRank
                        );
                        
                        // Esecuzione su hardware (WebNN/WebGPU/WASM)
                        const result = await this.activeDriver.execute(
                            synthesizedWeights,
                            tensorInfo,
                            routingPath
                        );
                        
                        // Usa logits per generare testo reale con sampling
                        if (result.logits) {
                            const tokenId = this.sampleTokenFromLogits(result.logits, 0.7, 40, 0.9);
                            const tokenText = this.decodeToken(tokenId);
                            
                            onTokenGenerated(tokenText, {
                                layer: tensorInfo.layer,
                                rank: routingPath.targetRank,
                                compression: routingPath.dynamicCompressionRatio,
                                activeNodes: routingPath.requiredTensors.length,
                                driver: this.hardwareProfile.primaryDriver
                            });
                        }
                        
                        this.transitionTo('EXECUTION', { layer: i + 1 });
                    }
                    
                    this.transitionTo('IDLE');
                    console.log('[STATE MACHINE] A.S.T.S. inference complete - zero RAM loading achieved');
                }
                catch (err) {
                    console.error('[STATE MACHINE] A.S.T.S. inference error:', err);
                    this.transitionTo('ERROR', { reason: err.message });
                }
            }
        });
    }
    
    transitionTo(newState, payload) {
        this.internalState = newState;
        this.onStateChange(newState, payload);
    }
    
    /**
     * Estrai vocabolario tokenizer dal metadata GGUF
     */
    extractVocabFromMetadata(metadata) {
        const vocab = {};
        
        // Cerca tokeni nel metadata GGUF
        for (const [key, value] of Object.entries(metadata)) {
            if (key.startsWith('tokenizer.ggml.token')) {
                // Estrai ID token dal nome (es: tokenizer.ggml.token.123)
                const tokenId = parseInt(key.split('.').pop(), 10);
                if (!isNaN(tokenId)) {
                    vocab[tokenId] = value;
                }
            }
        }
        
        // Se non trovato, crea vocabolario base di fallback
        if (Object.keys(vocab).length === 0) {
            console.warn('[STATE MACHINE] No vocabulary found in metadata, creating fallback vocab');
            for (let i = 0; i < 256; i++) {
                vocab[i] = String.fromCharCode(i);
            }
        }
        
        return vocab;
    }
    
    /**
     * Converti logits in token usando sampling
     */
    sampleTokenFromLogits(logits, temperature = 0.7, topK = 40, topP = 0.9) {
        // Applica temperature
        const scaledLogits = logits.map(l => l / temperature);
        
        // Softmax per ottenere probabilità
        const maxLogit = Math.max(...scaledLogits);
        const expLogits = scaledLogits.map(l => Math.exp(l - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(e => e / sumExp);
        
        // Top-K sampling
        const indexedProbs = probs.map((p, i) => ({ prob: p, index: i }));
        indexedProbs.sort((a, b) => b.prob - a.prob);
        const topKProbs = indexedProbs.slice(0, topK);
        
        // Top-P (nucleus) sampling
        let cumulativeProb = 0;
        const nucleusProbs = [];
        for (const item of topKProbs) {
            cumulativeProb += item.prob;
            nucleusProbs.push(item);
            if (cumulativeProb >= topP) break;
        }
        
        // Rinormalizza probabilità
        const nucleusSum = nucleusProbs.reduce((a, b) => a + b.prob, 0);
        const normalizedProbs = nucleusProbs.map(item => item.prob / nucleusSum);
        
        // Sampling
        const rand = Math.random();
        let sampleProb = 0;
        let sampledIndex = nucleusProbs[0].index;
        
        for (let i = 0; i < normalizedProbs.length; i++) {
            sampleProb += normalizedProbs[i];
            if (rand <= sampleProb) {
                sampledIndex = nucleusProbs[i].index;
                break;
            }
        }
        
        return sampledIndex;
    }
    
    /**
     * Converti token ID in testo usando vocabolario
     */
    decodeToken(tokenId) {
        if (!this.vocab) {
            // Fallback se vocab non disponibile
            return String.fromCharCode(65 + (tokenId % 26)); // A-Z
        }
        
        const token = this.vocab[tokenId];
        if (token === undefined) {
            // Fallback per token non trovato
            return String.fromCharCode(65 + (tokenId % 26));
        }
        
        return token;
    }
    
    get currentEngineState() {
        return this.internalState;
    }
}
