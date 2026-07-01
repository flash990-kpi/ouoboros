import { HardwareAuditor } from '../hw/hardwareAuditor';
describe('HardwareAuditor', () => {
    let auditor;
    beforeEach(() => {
        auditor = new HardwareAuditor();
    });
    test('should detect CPU capabilities', () => {
        const cpuInfo = auditor.auditCPU();
        expect(cpuInfo).toBeDefined();
        expect(cpuInfo.cores).toBeGreaterThan(0);
    });
    test('should detect memory information', () => {
        const memInfo = auditor.auditMemory();
        expect(memInfo).toBeDefined();
        expect(memInfo.total).toBeGreaterThan(0);
    });
    test('should detect GPU availability', () => {
        const gpuInfo = auditor.auditGPU();
        expect(gpuInfo).toBeDefined();
    });
    test('should provide hardware score', () => {
        const score = auditor.getHardwareScore();
        expect(score).toBeGreaterThanOrEqual(0);
        expect(score).toBeLessThanOrEqual(100);
    });
    test('should recommend optimizations', () => {
        const recommendations = auditor.getOptimizations();
        expect(Array.isArray(recommendations)).toBe(true);
    });
    test('should detect SIMD support', () => {
        const simdSupport = auditor.checkSIMDSupport();
        expect(typeof simdSupport).toBe('boolean');
    });
    test('should generate audit report', () => {
        const report = auditor.generateReport();
        expect(report).toBeDefined();
        expect(report.cpu).toBeDefined();
        expect(report.memory).toBeDefined();
        expect(report.gpu).toBeDefined();
    });
});
