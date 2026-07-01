export class OuroborosError extends Error {
  constructor(message: string, public code: string) {
    super(message);
    this.name = 'OuroborosError';
    Error.captureStackTrace(this, this.constructor);
  }
}

export class TopologyError extends OuroborosError {
  constructor(message: string) {
    super(message, 'TOPOLOGY_ERROR');
    this.name = 'TopologyError';
  }
}

export class WeightError extends OuroborosError {
  constructor(message: string) {
    super(message, 'WEIGHT_ERROR');
    this.name = 'WeightError';
  }
}

export class SparsityError extends OuroborosError {
  constructor(message: string) {
    super(message, 'SPARSITY_ERROR');
    this.name = 'SparsityError';
  }
}

export class HardwareError extends OuroborosError {
  constructor(message: string) {
    super(message, 'HARDWARE_ERROR');
    this.name = 'HardwareError';
  }
}

export class SchedulerError extends OuroborosError {
  constructor(message: string) {
    super(message, 'SCHEDULER_ERROR');
    this.name = 'SchedulerError';
  }
}

export class StateMachineError extends OuroborosError {
  constructor(message: string) {
    super(message, 'STATE_MACHINE_ERROR');
    this.name = 'StateMachineError';
  }
}

export class ValidationError extends OuroborosError {
  constructor(message: string, public validationErrors: string[]) {
    super(message, 'VALIDATION_ERROR');
    this.name = 'ValidationError';
  }
}

export function isOuroborosError(error: any): error is OuroborosError {
  return error instanceof OuroborosError;
}

export function handleError(error: unknown): void {
  if (isOuroborosError(error)) {
    console.error(`[${error.code}] ${error.message}`);
  } else if (error instanceof Error) {
    console.error(error.message);
  } else {
    console.error('Unknown error occurred');
  }
}
