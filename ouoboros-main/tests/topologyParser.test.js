import { TopologyParser } from '../asts/topologyParser';
describe('TopologyParser', () => {
    let parser;
    beforeEach(() => {
        parser = new TopologyParser();
    });
    test('should parse simple topology', () => {
        const topology = {
            layers: [
                { type: 'dense', units: 128 },
                { type: 'dense', units: 64 }
            ]
        };
        const result = parser.parse(topology);
        expect(result).toBeDefined();
        expect(result.layers).toHaveLength(2);
    });
    test('should validate topology structure', () => {
        const invalidTopology = { layers: [] };
        expect(() => parser.parse(invalidTopology)).toThrow();
    });
    test('should extract layer information', () => {
        const topology = {
            layers: [
                { type: 'conv2d', filters: 32, kernelSize: 3 }
            ]
        };
        const result = parser.parse(topology);
        expect(result.layers[0].type).toBe('conv2d');
    });
    test('should handle complex topologies', () => {
        const topology = {
            layers: [
                { type: 'conv2d', filters: 32, kernelSize: 3 },
                { type: 'maxpool2d', poolSize: 2 },
                { type: 'dense', units: 128 }
            ]
        };
        const result = parser.parse(topology);
        expect(result.layers).toHaveLength(3);
    });
    test('should compute topology statistics', () => {
        const topology = {
            layers: [
                { type: 'dense', units: 128 },
                { type: 'dense', units: 64 }
            ]
        };
        const result = parser.parse(topology);
        const stats = parser.getStatistics(result);
        expect(stats.totalLayers).toBe(2);
    });
});
