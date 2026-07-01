#!/usr/bin/env node

import { Command } from 'commander';
import { TopologyParser } from '../asts/topologyParser';
import { WeightSynthesizer } from '../asts/weightSynthesizer';
import { SparsityPredictor } from '../asts/sparsityPredictor';
import { logger } from '../ouroboros-core/utils/logger';
import * as fs from 'fs';

const program = new Command();

program
  .name('generate-topology')
  .description('Generate neural network topology with weights')
  .version('1.0.0')
  .option('-l, --layers <layers>', 'Number of layers', '3')
  .option('-u, --units <units>', 'Units per layer', '128')
  .option('-s, --sparsity <sparsity>', 'Target sparsity (0-1)', '0.5')
  .option('-o, --output <file>', 'Output file', 'topology.json')
  .parse(process.argv);

const options = program.opts();

function generateTopology() {
  try {
    const numLayers = parseInt(options.layers);
    const units = parseInt(options.units);
    const sparsity = parseFloat(options.sparsity);

    logger.info(`Generating topology with ${numLayers} layers, ${units} units each`);
    logger.info(`Target sparsity: ${sparsity}`);

    const topology = {
      layers: []
    };

    for (let i = 0; i < numLayers; i++) {
      topology.layers.push({
        type: 'dense',
        units: units,
        inputUnits: i === 0 ? 784 : units
      });
    }

    const parser = new TopologyParser();
    const parsed = parser.parse(topology);

    const synthesizer = new WeightSynthesizer();
    const predictor = new SparsityPredictor();

    for (const layer of parsed.layers) {
      const weights = synthesizer.synthesizeWithSparsity(layer, sparsity);
      layer.weights = weights;
      
      const actualSparsity = predictor.predict(layer);
      logger.debug(`Layer sparsity: ${actualSparsity.toFixed(3)}`);
    }

    const output = JSON.stringify(parsed, null, 2);
    fs.writeFileSync(options.output, output);

    logger.info(`Topology saved to ${options.output}`);
  } catch (error) {
    logger.error('Failed to generate topology:', error);
    process.exit(1);
  }
}

generateTopology();
