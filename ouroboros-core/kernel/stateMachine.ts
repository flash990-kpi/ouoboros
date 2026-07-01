import { EventEmitter } from 'events';

export enum InferenceState {
  IDLE = 'IDLE',
  ANALYZING = 'ANALYZING',
  SYNTHESIZING = 'SYNTHESIZING',
  EXECUTING = 'EXECUTING',
  COMPLETE = 'COMPLETE',
  ERROR = 'ERROR',
}

export interface InferenceContext {
  prompt: string;
  tokens: string[];
  inferenceTime: number;
  memoryUsed: number;
  hardwareUsed: 'NPU' | 'GPU' | 'CPU';
  topologyMap: Map<string, number[]>;
  activeLayers: Set<number>;
  metadata?: Record<string, unknown>;
}

export type StateTransition = {
  from: InferenceState;
  to: InferenceState;
  action?: () => Promise<void> | void;
  guard?: () => boolean;
};

export class InferenceStateMachine extends EventEmitter {
  private currentState: InferenceState = InferenceState.IDLE;
  private transitions: Map<string, StateTransition[]> = new Map();
  private context: Partial<InferenceContext> = {};
  private stateHistory: InferenceState[] = [];
  private startTime: number = 0;

  constructor() {
    super();
    this.registerDefaultTransitions();
  }

  private registerDefaultTransitions(): void {
    this.registerTransition({
      from: InferenceState.IDLE,
      to: InferenceState.ANALYZING,
      guard: () => !!this.context.prompt,
    });

    this.registerTransition({
      from: InferenceState.ANALYZING,
      to: InferenceState.SYNTHESIZING,
    });

    this.registerTransition({
      from: InferenceState.SYNTHESIZING,
      to: InferenceState.EXECUTING,
    });

    this.registerTransition({
      from: InferenceState.EXECUTING,
      to: InferenceState.COMPLETE,
    });

    this.registerTransition({
      from: InferenceState.EXECUTING,
      to: InferenceState.ERROR,
    });

    this.registerTransition({
      from: InferenceState.COMPLETE,
      to: InferenceState.IDLE,
    });

    this.registerTransition({
      from: InferenceState.ERROR,
      to: InferenceState.IDLE,
    });
  }

  registerTransition(transition: StateTransition): void {
    const key = `${transition.from}:${transition.to}`;
    if (!this.transitions.has(key)) {
      this.transitions.set(key, []);
    }
    this.transitions.get(key)!.push(transition);
  }

  async transition(toState: InferenceState): Promise<boolean> {
    const key = `${this.currentState}:${toState}`;
    const availableTransitions = this.transitions.get(key) || [];

    for (const transition of availableTransitions) {
      if (transition.guard && !transition.guard()) {
        this.emit('transition-blocked', {
          from: this.currentState,
          to: toState,
          reason: 'Guard condition failed',
        });
        return false;
      }
    }

    const transition = availableTransitions[0];
    if (!transition) {
      this.emit('invalid-transition', {
        from: this.currentState,
        to: toState,
      });
      return false;
    }

    const previousState = this.currentState;
    this.currentState = toState;
    this.stateHistory.push(toState);

    this.emit('state-changed', {
      from: previousState,
      to: toState,
      timestamp: Date.now(),
    });

    try {
      if (transition.action) {
        await transition.action();
      }
    } catch (error) {
      await this.transition(InferenceState.ERROR);
      this.emit('transition-error', {
        from: previousState,
        to: toState,
        error: error instanceof Error ? error.message : String(error),
      });
      return false;
    }

    return true;
  }

  setState(state: InferenceState): void {
    this.currentState = state;
    this.stateHistory.push(state);
    this.emit('state-set', { state, timestamp: Date.now() });
  }

  getState(): InferenceState {
    return this.currentState;
  }

  setContext(partial: Partial<InferenceContext>): void {
    this.context = { ...this.context, ...partial };
    this.emit('context-updated', this.context);
  }

  getContext(): Partial<InferenceContext> {
    return { ...this.context };
  }

  getHistory(): InferenceState[] {
    return [...this.stateHistory];
  }

  reset(): void {
    this.currentState = InferenceState.IDLE;
    this.context = {};
    this.stateHistory = [InferenceState.IDLE];
    this.startTime = 0;
    this.emit('state-machine-reset', { timestamp: Date.now() });
  }
}
