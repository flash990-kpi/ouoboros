import { Scheduler } from '../kernel/scheduler';
describe('Scheduler', () => {
    let scheduler;
    beforeEach(() => {
        scheduler = new Scheduler();
    });
    test('should initialize with default state', () => {
        expect(scheduler).toBeDefined();
    });
    test('should schedule tasks correctly', () => {
        const task = { id: '1', priority: 1, execute: jest.fn() };
        scheduler.schedule(task);
        expect(scheduler.getQueueLength()).toBeGreaterThan(0);
    });
    test('should execute tasks in priority order', async () => {
        const lowPriorityTask = { id: '2', priority: 2, execute: jest.fn() };
        const highPriorityTask = { id: '1', priority: 1, execute: jest.fn() };
        scheduler.schedule(lowPriorityTask);
        scheduler.schedule(highPriorityTask);
        await scheduler.processQueue();
        expect(highPriorityTask.execute).toHaveBeenCalledBefore(lowPriorityTask.execute);
    });
    test('should handle empty queue gracefully', async () => {
        await expect(scheduler.processQueue()).resolves.not.toThrow();
    });
    test('should cancel scheduled tasks', () => {
        const task = { id: '1', priority: 1, execute: jest.fn() };
        scheduler.schedule(task);
        scheduler.cancel('1');
        expect(scheduler.getQueueLength()).toBe(0);
    });
});
