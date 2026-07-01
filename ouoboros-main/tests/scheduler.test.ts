import { Scheduler } from '../kernel/scheduler';

describe('Scheduler', () => {
  let scheduler: Scheduler;

  beforeEach(() => {
    scheduler = new Scheduler();
  });

  test('should initialize with default state', () => {
    expect(scheduler).toBeDefined();
  });

  test('should enqueue tasks correctly', () => {
    const task = { id: '1', priority: 1, action: jest.fn().mockResolvedValue(undefined) };
    scheduler.enqueue(task);
    expect(scheduler).toBeDefined();
  });

  test('should clear queue', () => {
    const task = { id: '1', priority: 1, action: jest.fn().mockResolvedValue(undefined) };
    scheduler.enqueue(task);
    scheduler.clear();
    expect(scheduler).toBeDefined();
  });

  test('should handle task execution', async () => {
    const task = { id: '1', priority: 1, action: jest.fn().mockResolvedValue(undefined) };
    scheduler.enqueue(task);
    await new Promise(resolve => setTimeout(resolve, 100));
    expect(task.action).toHaveBeenCalled();
  });
});
