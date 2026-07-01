import { TopologyParser } from '../asts/topologyParser';
import { WeightSynthesizer } from '../asts/weightSynthesizer';
import { logger } from '../ouroboros-core/utils/logger';
import { metrics } from '../ouroboros-core/utils/metrics';

interface BatchConfig {
  batchSize: number;
  inputSize: number;
  topology: any;
}

class BatchInference {
  private topology: any;
  private batchSize: number;

  constructor(config: BatchConfig) {
    this.topology = config.topology;
    this.batchSize = config.batchSize;
  }

  async runBatch(inputs: number[][]): Promise<number[][]> {
    logger.info(`Running batch inference on ${inputs.length} inputs`);
    const startTime = performance.now();

    const results: number[][] = [];

    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      const output = await this.forward(input);
      results.push(output);
      
      if ((i + 1) % 10 === 0) {
        logger.info(`  Processed ${i + 1}/${inputs.length} inputs`);
      }
    }

    const endTime = performance.now();
    const totalTime = endTime - startTime;
    const throughput = inputs.length / (totalTime / 1000);

    metrics.record('batch_inference_time', totalTime);
    metrics.record('batch_throughput', throughput);

    logger.info(`Batch completed in ${totalTime.toFixed(2)}ms`);
    logger.info(`Throughput: ${throughput.toFixed(2)} inputs/sec`);

    return results;
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
    // ReLU activation
    return Math.max(0, x);
  }
}

async function main() {
  logger.info('🐍 Ouroboros Batch Inference Example');
  logger.info('=====================================\n');

  // Create topology
  const topology = {
    layers: [
      { type: 'dense', units: 128, inputUnits: 784 },
      { type: 'dense', units: 64, inputUnits: 128 },
      { type: 'dense', units: 10, inputUnits: 64 }
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

  // Create batch inference instance
  const batchSize = 100;
  const inputSize = 784;
  const batchInference = new BatchInference({
    batchSize,
    inputSize,
    topology: parsed
  });

  // Generate mock inputs
  logger.info(`Generating ${batchSize} mock inputs...`);
  const inputs: number[][] = [];
  for (let i = 0; i < batchSize; i++) {
    const input: number[] = [];
    for (let j = 0; j < inputSize; j++) {
      input.push(Math.random());
    }
    inputs.push(input);
  }
  logger.info('✓ Inputs generated\n');

  // Run batch inference
  logger.info('Running batch inference...\n');
  const results = await batchInference.runBatch(inputs);

  logger.info('\n=====================================');
  logger.info('✅ Batch inference completed!');
  logger.info(`Processed ${results.length} inputs`);
  logger.info(`Output shape: ${results[0].length}`);

  const timeSummary = metrics.getSummary('batch_inference_time');
  if (timeSummary) {
    logger.info(`Average inference time: ${timeSummary.mean.toFixed(2)}ms`);
  }

  const throughputSummary = metrics.getSummary('batch_throughput');
  if (throughputSummary) {
    logger.info(`Average throughput: ${throughputSummary.mean.toFixed(2)} inputs/sec`);
  }
}

main().catch(error => {
  logger.error('Batch inference failed:', error);
  process.exit(1);
});
