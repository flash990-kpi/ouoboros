import { OuroborosCore, OuroborosConfig, InferenceOptions } from './orchestrator';

export class OuroborosAPI {
  private core: OuroborosCore;
  private isReady: boolean = false;
  private eventListeners: Map<string, Set<Function>> = new Map();

  constructor(config: OuroborosConfig) {
    this.core = new OuroborosCore(config);
  }

  async init(): Promise<void> {
    try {
      await this.core.initialize();
      this.isReady = true;
      this.emit('ready', {
        hardware: this.core.getHardwareProfile(),
        topology: this.core.getTopology(),
      });
    } catch (error) {
      this.emit('error', { message: String(error) });
      throw error;
    }
  }

  async generate(prompt: string, maxTokens: number = 128): Promise<string> {
    if (!this.isReady) {
      throw new Error('API not initialized. Call init() first.');
    }

    return new Promise((resolve, reject) => {
      this.core
        .infer(
          { prompt, maxTokens },
          (token, isComplete) => {
            this.emit('token', { token, isComplete });
            if (isComplete) {
              resolve(token);
            }
          }
        )
        .catch(reject);
    });
  }

  on(event: string, listener: Function): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(listener);
  }

  off(event: string, listener: Function): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.delete(listener);
    }
  }

  private emit(event: string, data: any): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      for (const listener of listeners) {
        listener(data);
      }
    }
  }

  getStatus(): any {
    return {
      ready: this.isReady,
      hardware: this.core.getHardwareProfile(),
      topology: this.core.getTopology(),
      state: this.core.getState(),
      scheduler: this.core.getSchedulerStatus(),
    };
  }

  async dispose(): Promise<void> {
    await this.core.dispose();
    this.isReady = false;
  }
}

export default OuroborosAPI;
