export interface MetricData {
  name: string;
  value: number;
  timestamp: number;
  tags?: Record<string, string>;
}

export interface MetricSummary {
  count: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  stdDev: number;
}

export class MetricsCollector {
  private metrics: Map<string, MetricData[]> = new Map();

  record(name: string, value: number, tags?: Record<string, string>): void {
    const metric: MetricData = {
      name,
      value,
      timestamp: Date.now(),
      tags
    };

    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name)!.push(metric);
  }

  get(name: string): MetricData[] {
    return this.metrics.get(name) || [];
  }

  getSummary(name: string): MetricSummary | null {
    const data = this.get(name);
    if (data.length === 0) return null;

    const values = data.map(m => m.value).sort((a, b) => a - b);
    const sum = values.reduce((acc, v) => acc + v, 0);
    const mean = sum / values.length;
    const median = values[Math.floor(values.length / 2)];
    const variance = values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

    return {
      count: values.length,
      min: values[0],
      max: values[values.length - 1],
      mean,
      median,
      stdDev
    };
  }

  getAll(): Map<string, MetricData[]> {
    return this.metrics;
  }

  clear(name?: string): void {
    if (name) {
      this.metrics.delete(name);
    } else {
      this.metrics.clear();
    }
  }

  getNames(): string[] {
    return Array.from(this.metrics.keys());
  }
}

export const metrics = new MetricsCollector();
