import express from 'express';
import path from 'path';
import { OuroborosAPI } from './api';

const app = express();
const PORT = process.env.PORT || 3000;

let api: OuroborosAPI | null = null;

app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

app.post('/api/init', async (req, res) => {
  try {
    const { modelPath, topologyPath, maxMemory, thermalLimit } = req.body;

    api = new OuroborosAPI({
      modelPath: modelPath || './models/model.gguf',
      topologyPath,
      maxMemory,
      thermalLimit,
    });

    await api.init();

    res.json({
      success: true,
      status: api.getStatus(),
      message: 'A.S.T.S. Core initialized',
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: String(error),
    });
  }
});

app.post('/api/generate', async (req, res) => {
  try {
    if (!api) {
      return res.status(400).json({
        success: false,
        error: 'API not initialized. Call /api/init first.',
      });
    }

    const { prompt, maxTokens } = req.body;

    if (!prompt) {
      return res.status(400).json({
        success: false,
        error: 'Prompt is required',
      });
    }

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    api.on('token', (data: any) => {
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    });

    const result = await api.generate(prompt, maxTokens || 128);

    res.write('data: {"done": true}\n\n');
    res.end();
  } catch (error) {
    res.status(500).json({
      success: false,
      error: String(error),
    });
  }
});

app.get('/api/status', (req, res) => {
  if (!api) {
    return res.json({
      ready: false,
      message: 'API not initialized',
    });
  }

  res.json(api.getStatus());
});

app.post('/api/dispose', async (req, res) => {
  try {
    if (api) {
      await api.dispose();
      api = null;
    }

    res.json({
      success: true,
      message: 'A.S.T.S. Core disposed',
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: String(error),
    });
  }
});

app.listen(PORT, () => {
  console.log(`[Ouroboros Server] Running on http://localhost:${PORT}`);
  console.log(`[Ouroboros Server] A.S.T.S. Core ready for inference`);
});
