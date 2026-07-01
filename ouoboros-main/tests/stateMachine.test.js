import { StateMachine } from '../kernel/stateMachine';
describe('StateMachine', () => {
    let stateMachine;
    beforeEach(() => {
        stateMachine = new StateMachine();
    });
    test('should initialize with idle state', () => {
        expect(stateMachine.getCurrentState()).toBe('idle');
    });
    test('should transition to processing state', () => {
        stateMachine.transition('processing');
        expect(stateMachine.getCurrentState()).toBe('processing');
    });
    test('should reject invalid transitions', () => {
        stateMachine.transition('processing');
        expect(() => stateMachine.transition('invalid')).toThrow();
    });
    test('should allow valid state transitions', () => {
        stateMachine.transition('processing');
        stateMachine.transition('completed');
        expect(stateMachine.getCurrentState()).toBe('completed');
    });
    test('should trigger callbacks on state change', () => {
        const callback = jest.fn();
        stateMachine.onStateChange(callback);
        stateMachine.transition('processing');
        expect(callback).toHaveBeenCalledWith('processing');
    });
    test('should handle state history', () => {
        stateMachine.transition('processing');
        stateMachine.transition('completed');
        const history = stateMachine.getStateHistory();
        expect(history).toContain('processing');
        expect(history).toContain('completed');
    });
});
