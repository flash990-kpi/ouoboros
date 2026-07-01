import { WeightSynthesizer } from '../asts/weightSynthesizer';

describe('WeightSynthesizer', () => {
  let synthesizer: WeightSynthesizer;

  beforeEach(() => {
    synthesizer = new WeightSynthesizer();
  });

  test('should synthesize weights for dense layer', () => {
    const layer = { type: 'dense', units: 128, inputUnits: 64 };
    const weights = synthesizer.synthesize(layer);
    expect(weights).toBeDefined();
    expect(weights.length).toBe(128 * 64);
  });

  test('should synthesize weights for convolutional layer', () => {
    const layer = { type: 'conv2d', filters: 32, kernelSize: 3, inputChannels: 3 };
    const weights = synthesizer.synthesize(layer);
    expect(weights).toBeDefined();
    expect(weights.length).toBe(32 * 3 * 3 * 3);
  });

  test('should apply sparsity pattern', () => {
    const layer = { type: 'dense', units: 128, inputUnits: 64 };
    const sparsity = 0.5;
    const weights = synthesizer.synthesizeWithSparsity(layer, sparsity);
    const zeroCount = weights.filter(w => w === 0).length;
    const expectedZeros = Math.floor(weights.length * sparsity);
    expect(zeroCount).toBeGreaterThanOrEqual(expectedZeros - 10);
    expect(zeroCount).toBeLessThanOrEqual(expectedZeros + 10);
  });

  test('should validate weight dimensions', () => {
    const layer = { type: 'dense', units: 128, inputUnits: 64 };
    const weights = synthesizer.synthesize(layer);
    expect(synthesizer.validateDimensions(weights, layer)).toBe(true);
  });

  test('should normalize weights', () => {
    const layer = { type: 'dense', units: 128, inputUnits: 64 };
    const weights = synthesizer.synthesize(layer);
    const normalized = synthesizer.normalize(weights);
    const max = Math.max(...normalized);
    expect(max).toBeLessThanOrEqual(1);
  });
});
