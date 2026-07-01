import { OuroborosKernel } from '../kernel/stateMachine';

describe('OuroborosKernel', () => {
  test('should initialize with state change notifier', () => {
    const notifier = jest.fn();
    const kernel = new OuroborosKernel(notifier);
    expect(kernel).toBeDefined();
  });

  test('should have current engine state', () => {
    const notifier = jest.fn();
    const kernel = new OuroborosKernel(notifier);
    expect(kernel.currentEngineState).toBe('BOOTSTRAPPING');
  });
});
