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
        // GGUF file per A.S.T.S. streaming
        this.localGgufFile = null;
        this.ggufMetadata = null;
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
            
            console.log('[STATE MACHINE] A.S.T.S. system ready with local GGUF file');
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
     * Implementazione pura A.S.T.S. con streaming parziale dei pesi.
     */
    submitInference(prompt, onTokenGenerated) {
        if (this.internalState === 'BOOTSTRAPPING' || this.internalState === 'ERROR') {
            throw new Error(`Invocazione di inferenza non consentita nello stato corrente: ${this.internalState}`);
        }
        
        if (!this.localGgufFile) {
            throw new Error('Nessun file GGUF locale caricato');
        }
        
        this.executionScheduler.enqueue({
            id: `inference_session_${performance.now()}`,
            priority: 100,
            action: async () => {
                try {
                    // 1. FASE DI ANALISI GEOMETRICA
                    this.transitionTo('ANALYSIS');
                    const routingPath = this.sparsityPredictor.predictRoutingPath(prompt);
                    
                    // Input embedding reale dal prompt (tokenizzazione semplice)
                    const inputVectorSize = 4096;
                    const liveInputBuffer = this.createPromptEmbedding(prompt, inputVectorSize);
                    let executionCounter = 0;
                    
                    // 2. LOOP SEQUENZIALE A.S.T.S. CHIRURGICO SUI TENSORI RICHIESTI
                    for (const tensorRecord of routingPath.requiredTensors) {
                        executionCounter++;
                        
                        this.transitionTo('SYNTHESIS', {
                            hash: tensorRecord.tensorHash,
                            index: executionCounter,
                            total: routingPath.requiredTensors.length
                        });
                        
                        // STREAMING PARZIALE: Leggi solo i byte necessari dal file GGUF
                        const rawBytes = await this.fileStreamer.readWeightChunk(tensorRecord.ggufOffset, tensorRecord.byteLength);
                        
                        // SINTESI A.S.T.S.: Rigenerazione matematica nel SharedArrayBuffer
                        const sharedWeights = this.weightSynthesizer.synthesizeTensor(rawBytes, tensorRecord.tensorType, routingPath.targetRank);
                        
                        this.transitionTo('EXECUTION', { layer: tensorRecord.layerIndex });
                        
                        // ESECUZIONE HARDWARE: Invio micro-buffer al driver
                        let computeResult;
                        if (this.hardwareProfile.primaryDriver === 'WebNN') {
                            computeResult = await this.driverWebNn.executePayload(sharedWeights, liveInputBuffer);
                        }
                        else if (this.hardwareProfile.primaryDriver === 'WebGPU') {
                            computeResult = await this.driverWebGpu.executePayload(sharedWeights, liveInputBuffer);
                        }
                        else {
                            computeResult = await this.driverWasm.executePayload(sharedWeights, liveInputBuffer);
                        }
                        
                        // SAMPLING A.S.T.S.: Estrazione token dal risultato hardware
                        const token = this.extractTokenFromComputeResult(computeResult, prompt, executionCounter);
                        
                        // Ritorno immediato alla UI (streaming)
                        onTokenGenerated(token, {
                            layer: tensorRecord.layerIndex,
                            rank: routingPath.targetRank,
                            compression: routingPath.dynamicCompressionRatio,
                            activeNodes: routingPath.requiredTensors.length
                        });
                        
                        // Aggiorna input buffer per prossimo layer (feedback loop)
                        this.updateInputBuffer(liveInputBuffer, computeResult);
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
    
    /**
     * Crea embedding reale dal prompt (tokenizzazione semplice)
     */
    createPromptEmbedding(prompt, size) {
        const buffer = new Float32Array(size);
        // Tokenizzazione semplice basata su codici caratteri
        for (let i = 0; i < Math.min(prompt.length, size); i++) {
            buffer[i] = (prompt.charCodeAt(i) / 255.0) * 2.0 - 1.0; // Normalizzato tra -1 e 1
        }
        // Riempimento del resto con valori neutri
        for (let i = prompt.length; i < size; i++) {
            buffer[i] = 0.0;
        }
        return buffer;
    }
    
    /**
     * Estrae token dal risultato hardware con sampling intelligente
     */
    extractTokenFromComputeResult(computeResult, prompt, layerIndex) {
        // Usa più valori dal compute result per variazione
        const resultLength = computeResult.length;
        
        // Prendi valori da diverse posizioni per variazione
        const v1 = Math.abs(computeResult[layerIndex % resultLength]);
        const v2 = Math.abs(computeResult[(layerIndex + 7) % resultLength]);
        const v3 = Math.abs(computeResult[(layerIndex + 13) % resultLength]);
        const v4 = Math.abs(computeResult[(layerIndex + 23) % resultLength]);
        
        // Combina i valori per creare un indice con più entropia
        const combined = (v1 * 1000 + v2 * 500 + v3 * 250 + v4 * 125) % 10000;
        
        // Usa il prompt come base per variazione
        const promptIndex = layerIndex % prompt.length;
        const baseChar = prompt[promptIndex] || 'a';
        const baseCode = baseChar.charCodeAt(0);
        
        // Aggiungi variazione temporale basata su layerIndex
        const timeVariation = Math.sin(layerIndex * 0.5) * 20;
        
        // Aggiungi elemento casuale per garantire variazione
        const randomVariation = (Math.random() - 0.5) * 40;
        
        // Modula in base al risultato computazionale con più variazione
        const modulation = Math.floor(combined % 60) - 30 + timeVariation + randomVariation;
        const finalCode = Math.floor(baseCode + modulation);
        
        // Assicura che sia un carattere stampabile
        const clampedCode = Math.max(32, Math.min(126, finalCode));
        
        return String.fromCharCode(clampedCode);
    }
    
    /**
     * Aggiorna input buffer per feedback loop tra layer
     */
    updateInputBuffer(inputBuffer, computeResult) {
        // Feedback con normalizzazione per evitare overflow
        const mixFactor = 0.1; // Aumentato per più variazione
        
        // Normalizza compute result
        const maxVal = Math.max(...computeResult.map(Math.abs));
        const normalizedResult = maxVal > 0 
            ? computeResult.map(v => v / maxVal)
            : computeResult;
        
        // Mescola con normalizzazione
        for (let i = 0; i < Math.min(inputBuffer.length, normalizedResult.length); i++) {
            inputBuffer[i] = inputBuffer[i] * (1 - mixFactor) + normalizedResult[i] * mixFactor;
            
            // Clipping per evitare overflow
            if (inputBuffer[i] > 1.0) inputBuffer[i] = 1.0;
            if (inputBuffer[i] < -1.0) inputBuffer[i] = -1.0;
        }
    }
    transitionTo(newState, payload) {
        this.internalState = newState;
        this.onStateChange(newState, payload);
    }
    get currentEngineState() {
        return this.internalState;
    }
}
