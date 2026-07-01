import { TopologyParser } from '../asts/topologyParser';
import { WeightSynthesizer } from '../asts/weightSynthesizer';
import { SparsityPredictor } from '../asts/sparsityPredictor';
import { Scheduler } from '../kernel/scheduler';
import { StateMachine } from '../kernel/stateMachine';
import { logger } from '../ouroboros-core/utils/logger';
import { metrics } from '../ouroboros-core/utils/metrics';
import { Validators } from '../ouroboros-core/utils/validators';

async function main() {
  logger.info('🐍 Ouroboros Node.js Example');
  logger.info('================================\n');

  // 1. Create a simple topology
  logger.info('Step 1: Creating topology...');
  const topology = {
    layers: [
      { type: 'dense', units: 128, inputUnits: 784 },
      { type: 'dense', units: 64, inputUnits: 128 },
      { type: 'dense', units: 10, inputUnits: 64 }
    ]
  };

  // Validate topology
  const validation = Validators.validateTopology(topology);
  if (!validation.valid) {
    logger.error('Topology validation failed');
    process.exit(1);
  }
  logger.info('✓ Topology validated');

  // 2. Parse topology
  logger.info('\nStep 2: Parsing topology...');
  const parser = new TopologyParser();
  const parsed = parser.parse(topology);
  logger.info(`✓ Parsed ${parsed.layers.length} layers`);

  // 3. Synthesize weights with sparsity
  logger.info('\nStep 3: Synthesizing weights...');
  const synthesizer = new WeightSynthesizer();
  const predictor = new SparsityPredictor();

  for (const layer of parsed.layers) {
    const weights = synthesizer.synthesizeWithSparsity(layer, 0.5);
    layer.weights = weights;
    
    const sparsity = predictor.predict(layer);
    logger.info(`  Layer ${layer.type}: ${layer.units} units, sparsity: ${(sparsity * 100).toFixed(2)}%`);
    
    metrics.record('layer_sparsity', sparsity);
  }
  logger.info('✓ Weights synthesized');

  // 4. Set up state machine
  logger.info('\nStep 4: Setting up state machine...');
  const stateMachine = new StateMachine();
  stateMachine.onStateChange((state) => {
    logger.info(`  State changed: ${state}`);
  });
  stateMachine.transition('processing');
  logger.info('✓ State machine initialized');

  // 5. Schedule tasks
  logger.info('\nStep 5: Scheduling tasks...');
  const scheduler = new Scheduler();

  for (let i = 0; i < 5; i++) {
    scheduler.schedule({
      id: `task-${i}`,
      priority: Math.floor(Math.random() * 5),
      execute: () => {
        const start = performance.now();
        // Simulate work
        for (let j = 0; j < 1000000; j++) {
          Math.sqrt(j);
        }
        const end = performance.now();
        metrics.record('task_duration', end - start);
        logger.info(`  Task ${i} completed in ${(end - start).toFixed(2)}ms`);
      }
    });
  }
  logger.info('✓ Tasks scheduled');

  // 6. Process tasks
  logger.info('\nStep 6: Processing tasks...');
  await scheduler.processQueue();
  logger.info('✓ All tasks completed');

  // 7. Show metrics
  logger.info('\nStep 7: Metrics summary');
  const sparsitySummary = metrics.getSummary('layer_sparsity');
  if (sparsitySummary) {
    logger.info(`  Average sparsity: ${(sparsitySummary.mean * 100).toFixed(2)}%`);
  }

  const durationSummary = metrics.getSummary('task_duration');
  if (durationSummary) {
    logger.info(`  Average task duration: ${durationSummary.mean.toFixed(2)}ms`);
    logger.info(`  Total task time: ${durationSummary.sum?.toFixed(2)}ms`);
  }

  // 8. Complete state machine
  stateMachine.transition('completed');

  logger.info('\n================================');
  logger.info('✅ Example completed successfully!');
}

main().catch(error => {
  logger.error('Example failed:', error);
  process.exit(1);
});
