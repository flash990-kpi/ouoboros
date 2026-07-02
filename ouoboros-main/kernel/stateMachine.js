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
                // Generazione dinamica a caldo dall'header del GGUF se non è pre-indicizzato
                finalOuroBuffer = await this.fileStreamer.generateTopologyFromGguf();
            }
            else {
                finalOuroBuffer = ouroBuffer;
            }
            this.activeTopologyMap = this.topologyParser.parseIndex(finalOuroBuffer);
            this.sparsityPredictor = new SparsityPredictor(this.activeTopologyMap);
            this.transitionTo('IDLE', {
                driver: this.hardwareProfile.primaryDriver,
                tensors: this.activeTopologyMap.tensorCount,
                layers: this.activeTopologyMap.layerCount
            });
        }
        catch (error) {
            this.transitionTo('ERROR', { reason: error.message });
        }
    }
    /**
     * Inietta un prompt nel motore di navigazione geometrica A.S.T.S.
     */
    submitInference(prompt, onTokenGenerated) {
        console.log('[STATE MACHINE] submitInference called with prompt:', prompt);
        console.log('[STATE MACHINE] Current state:', this.internalState);
        
        if (this.internalState === 'BOOTSTRAPPING' || this.internalState === 'ERROR') {
            throw new Error(`Invocazione di inferenza non consentita nello stato corrente: ${this.internalState}`);
        }
        
        console.log('[STATE MACHINE] Enqueuing inference task...');
        this.executionScheduler.enqueue({
            id: `inference_session_${performance.now()}`,
            priority: 100,
            action: async () => {
                try {
                    console.log('[STATE MACHINE] Starting inference execution...');
                    // 1. FASE DI ANALISI GEOMETRICA
                    this.transitionTo('ANALYSIS');
                    console.log('[STATE MACHINE] Predicting routing path...');
                    const routingPath = this.sparsityPredictor.predictRoutingPath(prompt);
                    console.log('[STATE MACHINE] Routing path:', routingPath);
                    
                    // Array fittizio di input corrispondente all'embedding del token corrente (Dimensione standard 4096)
                    const inputVectorSize = 4096;
                    const liveInputBuffer = new Float32Array(inputVectorSize);
                    liveInputBuffer.fill(0.125); // Valore normalizzato statico di attivazione iniziale
                    let executionCounter = 0;
                    
                    console.log('[STATE MACHINE] Required tensors:', routingPath.requiredTensors.length);
                    
                    // 2. LOOP SEQUENZIALE A.S.T.S. CHIRURGICO SUI TENSORI RICHIESTI
                    for (const tensorRecord of routingPath.requiredTensors) {
                        executionCounter++;
                        console.log(`[STATE MACHINE] Processing tensor ${executionCounter}/${routingPath.requiredTensors.length}:`, tensorRecord);
                        
                        this.transitionTo('SYNTHESIS', {
                            hash: tensorRecord.tensorHash,
                            index: executionCounter,
                            total: routingPath.requiredTensors.length
                        });
                        
                        // Lettura mirata dei byte dall'offset calcolato
                        console.log('[STATE MACHINE] Reading weight chunk...');
                        const rawBytes = await this.fileStreamer.readWeightChunk(tensorRecord.ggufOffset, tensorRecord.byteLength);
                        console.log('[STATE MACHINE] Weight chunk read, size:', rawBytes.byteLength);
                        
                        // Rigenerazione matematica immediata nel SharedArrayBuffer
                        console.log('[STATE MACHINE] Synthesizing tensor...');
                        const sharedWeights = this.weightSynthesizer.synthesizeTensor(rawBytes, tensorRecord.tensorType, routingPath.targetRank);
                        console.log('[STATE MACHINE] Tensor synthesized');
                        
                        this.transitionTo('EXECUTION', { layer: tensorRecord.layerIndex });
                        
                        let computeResult;
                        // Dirottamento atomico al driver hardware corrispondente
                        console.log('[STATE MACHINE] Executing on driver:', this.hardwareProfile.primaryDriver);
                        if (this.hardwareProfile.primaryDriver === 'WebNN') {
                            computeResult = await this.driverWebNn.executePayload(sharedWeights, liveInputBuffer);
                        }
                        else if (this.hardwareProfile.primaryDriver === 'WebGPU') {
                            computeResult = await this.driverWebGpu.executePayload(sharedWeights, liveInputBuffer);
                        }
                        else {
                            computeResult = await this.driverWasm.executePayload(sharedWeights, liveInputBuffer);
                        }
                        
                        console.log('[STATE MACHINE] Compute result:', computeResult);
                        
                        // Campionamento deterministico dell'output del calcolo hardware per estrarre il token ASCII
                        const extractionIndex = Math.min(Math.floor(Math.abs(computeResult[0] * 100)), computeResult.length - 1);
                        const asciiCode = (Math.floor(Math.abs(computeResult[extractionIndex])) % 95) + 32;
                        const generatedToken = String.fromCharCode(asciiCode);
                        
                        console.log('[STATE MACHINE] Generated token:', generatedToken);
                        
                        // Ritorno immediato della metrica e del token alla UI WebSockets
                        onTokenGenerated(generatedToken, {
                            layer: tensorRecord.layerIndex,
                            rank: routingPath.targetRank,
                            compression: routingPath.dynamicCompressionRatio,
                            activeNodes: routingPath.requiredTensors.length
                        });
                    }
                    this.transitionTo('IDLE');
                    console.log('[STATE MACHINE] Inference completed successfully');
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
