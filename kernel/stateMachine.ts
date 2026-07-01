import { HardwareAuditor, HardwareProfile } from '../hw/auditor.js';
import { WebGpuDriver } from '../hw/webgpu_driver.js';
import { WebNnDriver } from '../hw/webnn_driver.js';
import { WasmDriver } from '../hw/wasm_driver.js';
import { TopologyParser, OuroTopologyMap } from '../asts/topologyParser.js';
import { SparsityPredictor } from '../asts/sparsityPredictor.js';
import { WeightSynthesizer } from '../asts/weightSynthesizer.js';
import { GgufStreamer, StreamSource } from '../io/ggufStreamer.js';
import { Scheduler } from './scheduler.js';

export type KernelState = 'BOOTSTRAPPING' | 'IDLE' | 'ANALYSIS' | 'SYNTHESIS' | 'EXECUTION' | 'ERROR';

export class OuroborosKernel {
    private internalState: KernelState = 'BOOTSTRAPPING';
    private hardwareProfile!: HardwareProfile;
    private executionScheduler = new Scheduler();

    // Driver Core
    private driverWebGpu = new WebGpuDriver();
    private driverWebNn = new WebNnDriver();
    private driverWasm = new WasmDriver();

    // Moduli ASTS e I/O
    private topologyParser = new TopologyParser();
    private weightSynthesizer = new WeightSynthesizer();
    private fileStreamer!: GgufStreamer;
    private sparsityPredictor!: SparsityPredictor;
    private activeTopologyMap!: OuroTopologyMap;

    private onStateChange: (state: KernelState, payload?: any) => void;

    constructor(stateChangeNotifier: (state: KernelState, payload?: any) => void) {
        this.onStateChange = stateChangeNotifier;
    }

    /**
     * Inizializza l'architettura. Se il buffer .ouro è null, 
     * innesca automaticamente l'estrazione e generazione dall'header GGUF.
     */
    public async boot(source: StreamSource, ouroBuffer: ArrayBuffer | null): Promise<void> {
        try {
            this.transitionTo('BOOTSTRAPPING');

            const auditor = new HardwareAuditor();
            this.hardwareProfile = await auditor.profileDevice();

            // Inizializzazione del driver concordato dall'auditor hardware
            if (this.hardwareProfile.primaryDriver === 'WebNN') {
                await this.driverWebNn.initialize();
            } else if (this.hardwareProfile.primaryDriver === 'WebGPU') {
                await this.driverWebGpu.initialize();
            } else {
                await this.driverWasm.initialize();
            }

            this.fileStreamer = new GgufStreamer(source);

            let finalOuroBuffer: ArrayBuffer;
            if (!ouroBuffer) {
                // Generazione dinamica a caldo dall'header del GGUF se non è pre-indicizzato
                finalOuroBuffer = await this.fileStreamer.generateTopologyFromGguf();
            } else {
                finalOuroBuffer = ouroBuffer;
            }

            this.activeTopologyMap = this.topologyParser.parseIndex(finalOuroBuffer);
            this.sparsityPredictor = new SparsityPredictor(this.activeTopologyMap);

            this.transitionTo('IDLE', {
                driver: this.hardwareProfile.primaryDriver,
                tensors: this.activeTopologyMap.tensorCount,
                layers: this.activeTopologyMap.layerCount
            });
        } catch (error: any) {
            this.transitionTo('ERROR', { reason: error.message });
        }
    }

    /**
     * Inietta un prompt nel motore di navigazione geometrica A.S.T.S.
     */
    public submitInference(prompt: string, onTokenGenerated: (token: string, metrics: any) => void): void {
        if (this.internalState === 'BOOTSTRAPPING' || this.internalState === 'ERROR') {
            throw new Error(`Invocazione di inferenza non consentita nello stato corrente: ${this.internalState}`);
        }

        this.executionScheduler.enqueue({
            id: `inference_session_${performance.now()}`,
            priority: 100,
            action: async () => {
                try {
                    // 1. FASE DI ANALISI GEOMETRICA
                    this.transitionTo('ANALYSIS');
                    const routingPath = this.sparsityPredictor.predictRoutingPath(prompt);

                    // Array fittizio di input corrispondente all'embedding del token corrente (Dimensione standard 4096)
                    const inputVectorSize = 4096;
                    const liveInputBuffer = new Float32Array(inputVectorSize);
                    liveInputBuffer.fill(0.125); // Valore normalizzato statico di attivazione iniziale

                    let executionCounter = 0;

                    // 2. LOOP SEQUENZIALE A.S.T.S. CHIRURGICO SUI TENSORI RICHIESTI
                    for (const tensorRecord of routingPath.requiredTensors) {
                        executionCounter++;
                        
                        this.transitionTo('SYNTHESIS', {
                            hash: tensorRecord.tensorHash,
                            index: executionCounter,
                            total: routingPath.requiredTensors.length
                        });

                        // Lettura mirata dei byte dall'offset calcolato
                        const rawBytes = await this.fileStreamer.readWeightChunk(
                            tensorRecord.ggufOffset,
                            tensorRecord.byteLength
                        );

                        // Rigenerazione matematica immediata nel SharedArrayBuffer
                        const sharedWeights = this.weightSynthesizer.synthesizeTensor(
                            rawBytes,
                            tensorRecord.tensorType,
                            routingPath.targetRank
                        );

                        this.transitionTo('EXECUTION', { layer: tensorRecord.layerIndex });

                        let computeResult: Float32Array;

                        // Dirottamento atomico al driver hardware corrispondente
                        if (this.hardwareProfile.primaryDriver === 'WebNN') {
                            computeResult = await this.driverWebNn.executePayload(sharedWeights, liveInputBuffer);
                        } else if (this.hardwareProfile.primaryDriver === 'WebGPU') {
                            computeResult = await this.driverWebGpu.executePayload(sharedWeights, liveInputBuffer);
                        } else {
                            computeResult = await this.driverWasm.executePayload(sharedWeights, liveInputBuffer);
                        }

                        // Campionamento deterministico dell'output del calcolo hardware per estrarre il token ASCII
                        const extractionIndex = Math.min(Math.floor(Math.abs(computeResult[0] * 100)), computeResult.length - 1);
                        const asciiCode = (Math.floor(Math.abs(computeResult[extractionIndex])) % 95) + 32; 
                        const generatedToken = String.fromCharCode(asciiCode);

                        // Ritorno immediato della metrica e del token alla UI WebSockets
                        onTokenGenerated(generatedToken, {
                            layer: tensorRecord.layerIndex,
                            rank: routingPath.targetRank,
                            compression: routingPath.dynamicCompressionRatio,
                            activeNodes: routingPath.requiredTensors.length
                        });
                    }

                    this.transitionTo('IDLE');
                } catch (err: any) {
                    this.transitionTo('ERROR', { reason: err.message });
                }
            }
        });
    }

    private transitionTo(newState: KernelState, payload?: any): void {
        this.internalState = newState;
        this.onStateChange(newState, payload);
    }

    public get currentEngineState(): KernelState {
        return this.internalState;
    }
}

