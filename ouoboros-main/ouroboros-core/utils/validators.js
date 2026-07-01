export class Validators {
    static validateTopology(topology) {
        const errors = [];
        if (!topology || typeof topology !== 'object') {
            errors.push('Topology must be an object');
            return { valid: false, errors };
        }
        if (!Array.isArray(topology.layers)) {
            errors.push('Topology must have a layers array');
            return { valid: false, errors };
        }
        if (topology.layers.length === 0) {
            errors.push('Topology must have at least one layer');
            return { valid: false, errors };
        }
        for (let i = 0; i < topology.layers.length; i++) {
            const layer = topology.layers[i];
            if (!layer.type) {
                errors.push(`Layer ${i} must have a type`);
            }
        }
        return {
            valid: errors.length === 0,
            errors
        };
    }
    static validateWeights(weights, expectedLength) {
        const errors = [];
        if (!Array.isArray(weights)) {
            errors.push('Weights must be an array');
            return { valid: false, errors };
        }
        if (weights.length !== expectedLength) {
            errors.push(`Weights length ${weights.length} does not match expected ${expectedLength}`);
        }
        for (let i = 0; i < weights.length; i++) {
            if (typeof weights[i] !== 'number' || isNaN(weights[i])) {
                errors.push(`Weight at index ${i} is not a valid number`);
            }
        }
        return {
            valid: errors.length === 0,
            errors
        };
    }
    static validateSparsity(sparsity) {
        const errors = [];
        if (typeof sparsity !== 'number') {
            errors.push('Sparsity must be a number');
            return { valid: false, errors };
        }
        if (sparsity < 0 || sparsity > 1) {
            errors.push('Sparsity must be between 0 and 1');
        }
        return {
            valid: errors.length === 0,
            errors
        };
    }
    static validateLayerConfig(config) {
        const errors = [];
        if (!config || typeof config !== 'object') {
            errors.push('Layer config must be an object');
            return { valid: false, errors };
        }
        if (!config.type) {
            errors.push('Layer config must have a type');
        }
        const validTypes = ['dense', 'conv2d', 'maxpool2d', 'flatten', 'dropout'];
        if (config.type && !validTypes.includes(config.type)) {
            errors.push(`Invalid layer type: ${config.type}`);
        }
        return {
            valid: errors.length === 0,
            errors
        };
    }
}
