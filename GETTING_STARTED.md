# A.S.T.S. Getting Started Guide

## Installation & Setup

### Prerequisites
- Node.js 18+
- npm or yarn
- Modern browser with WebGPU/WebNN support (or WASM fallback)

### 1. Clone & Install

```bash
git clone https://github.com/flash990-kpi/ouoboros.git
cd ouroboros
npm install
```

### 2. Prepare Model

Place your GGUF model in `models/` directory:

```bash
mkdir -p models
cp /path/to/your/model.gguf models/
```

### 3. Build

```bash
npm run build
```

### 4. Run Server

```bash
npm start
```

Server is ready at `http://localhost:3000`

## Web UI Usage

1. Open browser to `http://localhost:3000`
2. Click "Initialize" button
3. Enter your prompt in the textarea
4. Click "Generate"
5. Watch tokens stream in real-time

## Programmatic API

### Node.js Example

```typescript
import { OuroborosAPI } from './api';

async function main() {
  const api = new OuroborosAPI({
    modelPath: './models/model.gguf',
  });

  await api.init();
  console.log('Hardware:', api.getStatus().hardware);

  const result = await api.generate('Hello, world!', 50);
  console.log('Generated:', result);

  await api.dispose();
}

main();
```

### Browser Example

```html
<!DOCTYPE html>
<html>
<head>
  <script src="./dist/api.js"></script>
</head>
<body>
  <button onclick="generate()">Generate</button>
  <div id="output"></div>

  <script>
    const api = new OuroborosAPI({
      modelPath: '/models/model.gguf',
    });

    api.on('token', (data) => {
      document.getElementById('output').textContent += data.token;
    });

    async function generate() {
      await api.init();
      const result = await api.generate('What is AI?', 100);
      console.log('Done:', result);
    }
  </script>
</body>
</html>
```

## Performance Tuning

### Hardware Detection

A.S.T.S. automatically detects and uses:
1. **NPU (Neural Processing Unit)** if available (WebNN)
2. **GPU** if available (WebGPU)
3. **CPU** fallback (WASM+SIMD)

Check detected hardware:

```typescript
const status = api.getStatus();
console.log(status.hardware.primary.backend);  // NPU, GPU, or CPU
console.log(status.hardware.estimatedTokensPerSecond);
```

### Memory Optimization

Control memory usage:

```typescript
const api = new OuroborosAPI({
  modelPath: './models/model.gguf',
  maxMemory: 512 * 1024 * 1024,  // 512 MB limit
  thermalLimit: 'low',            // Conservative thermal profile
});
```

**Thermal Limit Effects:**
- `'low'`: Uses 60% of max memory, lowest power
- `'medium'`: Uses 80% of max memory, balanced
- `'high'`: Uses 100% of max memory, max performance (risk of throttling)

### Batch Processing

```typescript
const prompts = [
  'What is AI?',
  'Explain machine learning',
  'How does inference work?',
];

for (const prompt of prompts) {
  const result = await api.generate(prompt, 50);
  console.log(result);
}
```

## Troubleshooting

### "GGUF file not found"

```bash
# Check file exists
ls -lh models/model.gguf

# Verify it's a valid GGUF file
file models/model.gguf  # should say "data" or similar
```

### "No suitable hardware acceleration found"

Fallback to WASM is automatic. If you need specific backend:

```typescript
const status = api.getStatus();
if (status.hardware.primary.backend === 'CPU') {
  console.warn('No GPU/NPU detected, using CPU (slower)');
}
```

### "WebGPU not available"

- Ensure browser supports WebGPU (Chrome 113+, Edge, Firefox nightly)
- Or enable in chrome://flags (#enable-webgpu)
- WASM fallback will automatically activate

### "Out of memory"

Reduce `maxMemory` or `thermalLimit`:

```typescript
const api = new OuroborosAPI({
  modelPath: './models/model.gguf',
  maxMemory: 256 * 1024 * 1024,  // Reduce to 256 MB
  thermalLimit: 'low',
});
```

## Monitoring

### Real-time Status

```typescript
setInterval(() => {
  const status = api.getStatus();
  console.log({
    state: status.state,
    activeWorkers: status.scheduler.activeWorkers,
    taskQueue: status.scheduler.taskQueue,
  });
}, 1000);
```

### Event Logging

```typescript
api.on('ready', (data) => {
  console.log('Ready with hardware:', data.hardware);
});

api.on('token', (data) => {
  console.log('Token:', data.token);
});

api.on('error', (data) => {
  console.error('Error:', data.message);
});
```

## Advanced Usage

### Custom Topology

If you have a pre-computed `.ouro` file:

```typescript
const api = new OuroborosAPI({
  modelPath: './models/model.gguf',
  topologyPath: './models/model.ouro',  // Specify custom topology
});
```

### Temperature & Sampling

```typescript
const result = await api.generate('Write a story:', 200, {
  temperature: 1.2,  // Higher = more creative
  topP: 0.95,        // Nucleus sampling
});
```

## Next Steps

- Read [README.md](./README.md) for architecture details
- Check [API documentation](./ARCHITECTURE.md) for deep dives
- Explore `ouroboros-core/` source for implementation details

---

**Questions?** Open an issue on GitHub!
