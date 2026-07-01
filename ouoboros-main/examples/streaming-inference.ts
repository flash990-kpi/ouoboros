import { TopologyParser } from '../asts/topologyParser';
import { WeightSynthesizer } from '../asts/weightSynthesizer';
import { Scheduler } from '../kernel/scheduler';
import { StateMachine } from '../kernel/stateMachine';
import { logger } from '../ouroboros-core/utils/logger';
import { metrics } from '../ouroboros-core/utils/metrics';

interface StreamingConfig {
  bufferSize: number;
  inputSize: number;
  topology: any;
}

class StreamingInference {
  private topology: any;
  private bufferSize: number;
  private buffer: number[][];
  private scheduler: Scheduler;
  private stateMachine: StateMachine;

  constructor(config: StreamingConfig) {
    this.topology = config.topology;
    this.bufferSize = config.bufferSize;
    this.buffer = [];
    this.scheduler = new Scheduler();
    this.stateMachine = new StateMachine();
  }

  async start(): Promise<void> {
    logger.info('Starting streaming inference...');
    this.stateMachine.transition('processing');
    
    // Schedule buffer processing task
    this.scheduler.schedule({
      id: 'buffer-processor',
      priority: 1,
      execute: async () => {
        await this.processBuffer();
      }
    });

    // Start continuous processing
    setInterval(() => {
      if (this.buffer.length >= this.bufferSize) {
        this.scheduler.schedule({
          id: `process-${Date.now()}`,
          priority: 1,
          execute: async () => {
            await this.processBuffer();
          }
        });
      }
    }, 100);
  }

  async addInput(input: number[]): Promise<void> {
    this.buffer.push(input);
    
    if (this.buffer.length > this.bufferSize) {
      this.buffer.shift(); // Remove oldest
    }

    metrics.record('buffer_size', this.buffer.length);
  }

  private async processBuffer(): Promise<void> {
    if (this.buffer.length === 0) return;

    const startTime = performance.now();
    const results: number[][] = [];

    for (const input of this.buffer) {
      const output = await this.forward(input);
      results.push(output);
    }

    const endTime = performance.now();
    const processingTime = endTime - startTime;

    metrics.record('streaming_processing_time', processingTime);
    metrics.record('streaming_batch_size', this.buffer.length);

    logger.debug(`Processed ${this.buffer.length} inputs in ${processingTime.toFixed(2)}ms`);

    // Clear buffer after processing
    this.buffer = [];
  }

  private async forward(input: number[]): Promise<number[]> {
    let current = input;

    for (const layer of this.topology.layers) {
      current = this.applyLayer(current, layer);
    }

    return current;
  }

  private applyLayer(input: number[], layer: any): number[] {
    const weights = layer.weights || [];
    const units = layer.units;
    const inputUnits = layer.inputUnits || input.length;

    const output: number[] = [];

    for (let i = 0; i < units; i++) {
      let sum = 0;
      for (let j = 0; j < inputUnits; j++) {
        const weightIndex = i * inputUnits + j;
        const weight = weights[weightIndex] || 0;
        sum += input[j] * weight;
      }
      output.push(this.activation(sum));
    }

    return output;
  }

  private activation(x: number): number {
    return Math.max(0, x);
  }

  getStats(): any {
    return {
      bufferSize: this.buffer.length,
      state: this.stateMachine.getCurrentState(),
      processingTime: metrics.getSummary('streaming_processing_time'),
      batchSize: metrics.getSummary('streaming_batch_size')
    };
  }
}

async function main() {
  logger.info('🐍 Ouroboros Streaming Inference Example');
  logger.info('=======================================\n');

  // Create topology
  const topology = {
    layers: [
      { type: 'dense', units: 64, inputUnits: 784 },
      { type: 'dense', units: 32, inputUnits: 64 },
      { type: 'dense', units: 10, inputUnits: 32 }
    ]
  };

  // Synthesize weights
  logger.info('Synthesizing weights...');
  const synthesizer = new WeightSynthesizer();
  const parser = new TopologyParser();
  const parsed = parser.parse(topology);

  for (const layer of parsed.layers) {
    const weights = synthesizer.synthesizeWithSparsity(layer, 0.5);
    layer.weights = weights;
  }
  logger.info('✓ Weights synthesized\n');

  // Create streaming inference instance
  const bufferSize = 10;
  const inputSize = 784;
  const streaming = new StreamingInference({
    bufferSize,
    inputSize,
    topology: parsed
  });

  // Start streaming
  await streaming.start();
  logger.info('✓ Streaming started\n');

  // Simulate incoming data stream
  logger.info('Simulating data stream...');
  let totalInputs = 0;
  const maxInputs = 100;

  const streamInterval = setInterval(() => {
    if (totalInputs >= maxInputs) {
      clearInterval(streamInterval);
      logger.info('\nStream simulation completed');
      
      // Show final stats
      const stats = streaming.getStats();
      logger.info('\nFinal Statistics:');
      logger.info(`  Buffer size: ${stats.bufferSize}`);
      logger.info(`  State: ${stats.state}`);
      
      if (stats.processingTime) {
        logger.info(`  Avg processing time: ${stats.processingTime.mean.toFixed(2)}ms`);
      }
      
      if (stats.batchSize) {
        logger.info(`  Avg batch size: ${stats.batchSize.mean.toFixed(2)}`);
      }
      
      logger.info('\n=======================================');
      logger.info('✅ Streaming inference example completed!');
      process.exit(0);
      return;
    }

    // Generate mock input
    const input: number[] = [];
    for (let i = 0; i < inputSize; i++) {
      input.push(Math.random());
    }

    streaming.addInput(input);
    totalInputs++;

    if (totalInputs % 10 === 0) {
      logger.info(`  Streamed ${totalInputs}/${maxInputs} inputs`);
    }
  }, 50);
}

main().catch(error => {
  logger.error('Streaming inference failed:', error);
  process.exit(1);
});
