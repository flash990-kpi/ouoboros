import { SparsityPredictor } from '../asts/sparsityPredictor';

describe('SparsityPredictor', () => {
  let predictor: SparsityPredictor;

  beforeEach(() => {
    predictor = new SparsityPredictor();
  });

  test('should predict sparsity for dense layers', () => {
    const layer = { type: 'dense', units: 128, weights: new Array(128 * 128).fill(0.1) };
    const sparsity = predictor.predict(layer);
    expect(sparsity).toBeGreaterThanOrEqual(0);
    expect(sparsity).toBeLessThanOrEqual(1);
  });

  test('should predict sparsity for convolutional layers', () => {
    const layer = { type: 'conv2d', filters: 32, kernelSize: 3, weights: new Array(32 * 3 * 3).fill(0.1) };
    const sparsity = predictor.predict(layer);
    expect(sparsity).toBeGreaterThanOrEqual(0);
    expect(sparsity).toBeLessThanOrEqual(1);
  });

  test('should handle zero weights', () => {
    const layer = { type: 'dense', units: 64, weights: new Array(64 * 64).fill(0) };
    const sparsity = predictor.predict(layer);
    expect(sparsity).toBe(1);
  });

  test('should handle non-zero weights', () => {
    const layer = { type: 'dense', units: 64, weights: new Array(64 * 64).fill(1) };
    const sparsity = predictor.predict(layer);
    expect(sparsity).toBe(0);
  });

  test('should provide confidence score', () => {
    const layer = { type: 'dense', units: 128, weights: new Array(128 * 128).fill(0.1) };
    const result = predictor.predictWithConfidence(layer);
    expect(result.sparsity).toBeDefined();
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });
});
