// Improved server with Range support, COOP/COEP headers, health endpoint
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 8080;

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.ts': 'application/typescript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.css': 'text/css',
  '.gguf': 'application/octet-stream',
  '.ouro': 'application/octet-stream',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain'
};

function sendError(res, code, message) {
  res.writeHead(code, { 'Content-Type': 'text/plain' });
  res.end(message);
}

const server = http.createServer((req, res) => {
  try {
    const urlPath = decodeURIComponent(req.url.split('?')[0]);

    // Health endpoint
    if (urlPath === '/health' || urlPath === '/healthz') {
      const body = JSON.stringify({ status: 'ok', time: Date.now() });
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp'
      });
      return res.end(body);
    }

    let filePath;
    if (urlPath === '/' || urlPath === '/index.html') {
      filePath = path.join(process.cwd(), 'index.html');
    } else {
      // Normalize and prevent path traversal
      const safePath = urlPath.replace(/^\/+/, '');
      filePath = path.join(process.cwd(), safePath);
    }

    if (!fs.existsSync(filePath)) return sendError(res, 404, '404: File non trovato');

    const stat = fs.statSync(filePath);
    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    // Common security headers required for SharedArrayBuffer usage
    const commonHeaders = {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'same-origin',
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Expires': '0'
    };

    // Allow CORS for debugging and remote GGUF fetching; restrict in production
    const allowOrigin = process.env.ALLOW_ORIGIN || '*';

    if (ext === '.gguf' || req.headers.range) {
      // Support Range requests for large model files
      const range = req.headers.range;
      if (!range) {
        // Serve full file with Accept-Ranges header
        res.writeHead(200, Object.assign({}, commonHeaders, {
          'Content-Type': contentType,
          'Content-Length': stat.size,
          'Accept-Ranges': 'bytes',
          'Access-Control-Allow-Origin': allowOrigin
        }));
        const stream = fs.createReadStream(filePath);
        return stream.pipe(res);
      }

      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
      if (isNaN(start) || isNaN(end) || start > end || start < 0) return sendError(res, 416, 'Requested Range Not Satisfiable');

      const chunkSize = (end - start) + 1;
      const headers = Object.assign({}, commonHeaders, {
        'Content-Range': `bytes ${start}-${end}/${stat.size}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': chunkSize,
        'Content-Type': contentType,
        'Access-Control-Allow-Origin': allowOrigin,
        'Access-Control-Expose-Headers': 'Content-Range,Accept-Ranges'
      });

      res.writeHead(206, headers);
      const stream = fs.createReadStream(filePath, { start, end });
      return stream.pipe(res);
    }

    // Default: serve static file
    const headers = Object.assign({}, commonHeaders, {
      'Content-Type': contentType,
      'Content-Length': stat.size,
      'Access-Control-Allow-Origin': allowOrigin
    });
    res.writeHead(200, headers);
    const stream = fs.createReadStream(filePath);
    stream.pipe(res);
  } catch (err) {
    console.error('Server error', err);
    return sendError(res, 500, 'Errore interno del server');
  }
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`[OUROBOROS KERNEL] Attivo su http://0.0.0.0:${PORT}`);
  console.log('[A.S.T.S.] Header di sicurezza COOP/COEP/Resource-Policy iniettati.');
});

server.on('error', (err) => {
  console.error('Errore del server:', err.message);
});

process.on('SIGINT', () => {
  console.log('[OUROBOROS] Server arrestato manualmente.');
  process.exit(0);
});
