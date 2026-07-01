export class OuroborosError extends Error {
    constructor(message, code) {
        super(message);
        this.code = code;
        this.name = 'OuroborosError';
        Error.captureStackTrace(this, this.constructor);
    }
}
export class TopologyError extends OuroborosError {
    constructor(message) {
        super(message, 'TOPOLOGY_ERROR');
        this.name = 'TopologyError';
    }
}
export class WeightError extends OuroborosError {
    constructor(message) {
        super(message, 'WEIGHT_ERROR');
        this.name = 'WeightError';
    }
}
export class SparsityError extends OuroborosError {
    constructor(message) {
        super(message, 'SPARSITY_ERROR');
        this.name = 'SparsityError';
    }
}
export class HardwareError extends OuroborosError {
    constructor(message) {
        super(message, 'HARDWARE_ERROR');
        this.name = 'HardwareError';
    }
}
export class SchedulerError extends OuroborosError {
    constructor(message) {
        super(message, 'SCHEDULER_ERROR');
        this.name = 'SchedulerError';
    }
}
export class StateMachineError extends OuroborosError {
    constructor(message) {
        super(message, 'STATE_MACHINE_ERROR');
        this.name = 'StateMachineError';
    }
}
export class ValidationError extends OuroborosError {
    constructor(message, validationErrors) {
        super(message, 'VALIDATION_ERROR');
        this.validationErrors = validationErrors;
        this.name = 'ValidationError';
    }
}
export function isOuroborosError(error) {
    return error instanceof OuroborosError;
}
export function handleError(error) {
    if (isOuroborosError(error)) {
        console.error(`[${error.code}] ${error.message}`);
    }
    else if (error instanceof Error) {
        console.error(error.message);
    }
    else {
        console.error('Unknown error occurred');
    }
}
