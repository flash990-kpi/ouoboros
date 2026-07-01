#!/usr/bin/env node

import { Command } from 'commander';
import { Scheduler } from '../kernel/scheduler';
import { HardwareAuditor } from '../hw/hardwareAuditor';
import { metrics } from '../ouroboros-core/utils/metrics';
import { logger } from '../ouroboros-core/utils/logger';

const program = new Command();

program
  .name('benchmark')
  .description('Benchmark Ouroboros performance')
  .version('1.0.0')
  .option('-i, --iterations <iterations>', 'Number of iterations', '1000')
  .option('-t, --tasks <tasks>', 'Number of concurrent tasks', '10')
  .parse(process.argv);

const options = program.opts();

async function benchmark() {
  try {
    const iterations = parseInt(options.iterations);
    const tasks = parseInt(options.tasks);

    logger.info(`Starting benchmark: ${iterations} iterations, ${tasks} concurrent tasks`);

    const auditor = new HardwareAuditor();
    const hwReport =auditor.generateReport();
    
    logger.info('Hardware configuration:');
    logger.info(`  CPU cores: ${hwReport.cpu.cores}`);
    logger.info(`  Memory: ${hwReport.memory.total} MB`);
    logger.info(`  GPU: ${hwReport.gpu.available ? 'Available' : 'Not available'}`);

    const scheduler = new Scheduler();

    logger.info('Running scheduler benchmark...');
    const startTime = Date.now();

    for (let i = 0; i < iterations; i++) {
      const task = {
        id: `task-${i}`,
        priority: Math.floor(Math.random() * 10),
        execute: () => {
          const start = performance.now();
          for (let j = 0; j < 1000; j++) {
            Math.sqrt(j);
          }
          const end = performance.now();
          metrics.record('task_execution_time', end - start);
        }
      };
      scheduler.schedule(task);
    }

    await scheduler.processQueue();
    const endTime = Date.now();

    const totalTime = endTime - startTime;
    const throughput = iterations / (totalTime / 1000);

    logger.info('Benchmark results:');
    logger.info(`  Total time: ${totalTime} ms`);
    logger.info(`  Throughput: ${throughput.toFixed(2)} tasks/sec`);

    const execSummary = metrics.getSummary('task_execution_time');
    if (execSummary) {
      logger.info('Task execution time statistics:');
      logger.info(`  Mean: ${execSummary.mean.toFixed(3)} ms`);
      logger.info(`  Median: ${execSummary.median.toFixed(3)} ms`);
      logger.info(`  Min: ${execSummary.min.toFixed(3)} ms`);
      logger.info(`  Max: ${execSummary.max.toFixed(3)} ms`);
      logger.info(`  Std Dev: ${execSummary.stdDev.toFixed(3)} ms`);
    }

    const hwScore = auditor.getHardwareScore();
    logger.info(`Hardware score: ${hwScore}/100`);

    logger.info('Benchmark completed successfully');
  } catch (error) {
    logger.error('Benchmark failed:', error);
    process.exit(1);
  }
}

benchmark();
