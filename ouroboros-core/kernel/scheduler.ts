import { EventEmitter } from 'events';

interface Task {
  id: string;
  priority: number;
  execute: () => Promise<void>;
  callbacks: {
    onStart?: () => void;
    onComplete?: () => void;
    onError?: (error: Error) => void;
  };
}

interface WorkerThread {
  id: string;
  active: boolean;
  currentTask: Task | null;
  completedTasks: number;
}

export class Scheduler extends EventEmitter {
  private tasks: Map<string, Task> = new Map();
  private queue: Task[] = [];
  private workers: WorkerThread[] = [];
  private maxWorkers: number;
  private running: boolean = false;

  constructor(maxWorkers: number = navigator.hardwareConcurrency || 4) {
    super();
    this.maxWorkers = Math.min(maxWorkers, navigator.hardwareConcurrency || 4);
    this.initializeWorkers();
  }

  private initializeWorkers(): void {
    for (let i = 0; i < this.maxWorkers; i++) {
      this.workers.push({
        id: `worker-${i}`,
        active: false,
        currentTask: null,
        completedTasks: 0,
      });
    }
  }

  enqueueTask(
    id: string,
    execute: () => Promise<void>,
    priority: number = 0,
    callbacks?: Task['callbacks']
  ): void {
    const task: Task = {
      id,
      priority,
      execute,
      callbacks: callbacks || {},
    };

    this.tasks.set(id, task);
    this.queue.push(task);
    this.queue.sort((a, b) => b.priority - a.priority);
    this.emit('task-enqueued', { taskId: id, priority });

    if (!this.running) {
      this.start();
    }
  }

  private start(): void {
    if (this.running) return;
    this.running = true;
    this.processQueue();
  }

  private async processQueue(): Promise<void> {
    while (this.queue.length > 0 || this.workers.some((w) => w.active)) {
      const availableWorker = this.workers.find((w) => !w.active);

      if (availableWorker && this.queue.length > 0) {
        const task = this.queue.shift();
        if (task) {
          await this.executeTaskOnWorker(availableWorker, task);
        }
      } else {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
    }
    this.running = false;
  }

  private async executeTaskOnWorker(
    worker: WorkerThread,
    task: Task
  ): Promise<void> {
    worker.active = true;
    worker.currentTask = task;
    this.emit('task-started', { workerId: worker.id, taskId: task.id });

    try {
      task.callbacks.onStart?.();
      await task.execute();
      task.callbacks.onComplete?.();
      worker.completedTasks++;
      this.emit('task-completed', { workerId: worker.id, taskId: task.id });
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      task.callbacks.onError?.(err);
      this.emit('task-error', { workerId: worker.id, taskId: task.id, error: err.message });
    } finally {
      worker.active = false;
      worker.currentTask = null;
      this.tasks.delete(task.id);
    }
  }

  getStatus() {
    return {
      totalTasks: this.tasks.size,
      queueLength: this.queue.length,
      activeWorkers: this.workers.filter((w) => w.active).length,
      workers: this.workers.map((w) => ({
        id: w.id,
        active: w.active,
        completedTasks: w.completedTasks,
        currentTaskId: w.currentTask?.id || null,
      })),
    };
  }
}
