#!/usr/bin/env node
import { Command } from 'commander';
import { TopologyParser } from '../asts/topologyParser';
import { Validators } from '../ouroboros-core/utils/validators';
import { logger } from '../ouroboros-core/utils/logger';
import * as fs from 'fs';
const program = new Command();
program
    .name('validate-model')
    .description('Validate neural network model topology')
    .version('1.0.0')
    .option('-i, --input <file>', 'Input model file', 'model.json')
    .parse(process.argv);
const options = program.opts();
function validateModel() {
    try {
        logger.info(`Validating model: ${options.input}`);
        if (!fs.existsSync(options.input)) {
            logger.error(`File not found: ${options.input}`);
            process.exit(1);
        }
        const content = fs.readFileSync(options.input, 'utf-8');
        const model = JSON.parse(content);
        const topologyValidation = Validators.validateTopology(model);
        if (!topologyValidation.valid) {
            logger.error('Topology validation failed:');
            topologyValidation.errors.forEach(err => logger.error(`  - ${err}`));
            process.exit(1);
        }
        logger.info('Topology validation: PASSED');
        const parser = new TopologyParser();
        const parsed = parser.parse(model);
        logger.info(`Parsed ${parsed.layers.length} layers`);
        for (let i = 0; i < parsed.layers.length; i++) {
            const layer = parsed.layers[i];
            const layerValidation = Validators.validateLayerConfig(layer);
            if (!layerValidation.valid) {
                logger.error(`Layer ${i} validation failed:`);
                layerValidation.errors.forEach(err => logger.error(`  - ${err}`));
                process.exit(1);
            }
            if (layer.weights) {
                const expectedLength = layer.units * (layer.inputUnits || layer.units);
                const weightsValidation = Validators.validateWeights(layer.weights, expectedLength);
                if (!weightsValidation.valid) {
                    logger.error(`Layer ${i} weights validation failed:`);
                    weightsValidation.errors.forEach(err => logger.error(`  - ${err}`));
                    process.exit(1);
                }
            }
        }
        logger.info('All validations: PASSED');
        logger.info('Model is valid!');
    }
    catch (error) {
        logger.error('Validation failed:', error);
        process.exit(1);
    }
}
validateModel();
