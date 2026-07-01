import { HardwareBackend, HardwareCapabilities } from './auditor';

export class WebNNDriver {
  private context: any = null;
  private capabilities: HardwareCapabilities;
  private isInitialized: boolean = false;

  constructor(capabilities: HardwareCapabilities) {
    this.capabilities = capabilities;
  }

  async initialize(): Promise<void> {
    try {
      if (!('ml' in navigator)) {
        throw new Error('WebNN not available');
      }

      this.context = (navigator as any).ml.createContextSync();
      this.isInitialized = true;
      console.log('WebNN driver initialized');
    } catch (error) {
      throw new Error(`WebNN initialization failed: ${error}`);
    }
  }

  async compute(
    weights: Float32Array,
    inputs: Float32Array,
    config: Record<string, unknown>
  ): Promise<Float32Array> {
    if (!this.isInitialized) {
      throw new Error('WebNN driver not initialized');
    }

    try {
      const builder = this.context.graph.builder();

      const weightsOperand = builder.constant(
        {
          dataType: 'float32',
          dimensions: [weights.length],
        },
        weights
      );

      const inputOperand = builder.input('input', {
        dataType: 'float32',
        dimensions: [inputs.length],
      });

      const outputOperand = builder.matmul(inputOperand, weightsOperand);
      const graph = builder.build({ output: outputOperand });

      const inputs_obj = new TypedArray('float32', inputs);
      const result = this.context.compute(graph, { input: inputs_obj });

      return new Float32Array(result.output);
    } catch (error) {
      throw new Error(`WebNN compute failed: ${error}`);
    }
  }

  async computeMatMul(
    matrixA: Float32Array,
    matrixB: Float32Array,
    dimsA: [number, number],
    dimsB: [number, number]
  ): Promise<Float32Array> {
    if (!this.isInitialized) {
      throw new Error('WebNN driver not initialized');
    }

    try {
      const builder = this.context.graph.builder();

      const aOperand = builder.constant(
        { dataType: 'float32', dimensions: dimsA },
        matrixA
      );

      const bOperand = builder.constant(
        { dataType: 'float32', dimensions: dimsB },
        matrixB
      );

      const resultOperand = builder.matmul(aOperand, bOperand);
      const graph = builder.build({ output: resultOperand });
      const result = this.context.compute(graph, {});

      return new Float32Array(result.output);
    } catch (error) {
      throw new Error(`WebNN MatMul failed: ${error}`);
    }
  }

  getCapabilities(): HardwareCapabilities {
    return this.capabilities;
  }

  isReady(): boolean {
    return this.isInitialized;
  }

  async dispose(): Promise<void> {
    try {
      if (this.context) {
        this.context.close?.();
        this.context = null;
        this.isInitialized = false;
      }
    } catch (error) {
      console.error('Error disposing WebNN driver:', error);
    }
  }
}
