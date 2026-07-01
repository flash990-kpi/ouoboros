// /ouroboros-core/server.js
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 8080;

const MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.mjs': 'text/javascript',
    '.ts': 'text/typescript',
    '.wasm': 'application/wasm',
    '.json': 'application/json',
    '.css': 'text/css',
    '.gguf': 'application/octet-stream',
    '.ouro': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.txt': 'text/plain'
};

const server = http.createServer((req, res) => {
    let urlPath = req.url.split('?')[0];

    console.log(`[SERVER] Request: ${req.method} ${urlPath}`);

    // Se la richiesta è per la root, serviamo index.html dalla root
    let filePath;
    if (urlPath === '/') {
        filePath = './index.html';
    } 
    // Se la richiesta è per node_modules, serviamo dalla cartella node_modules locale
    else if (urlPath.startsWith('/node_modules/')) {
        filePath = '.' + urlPath; // ad esempio /node_modules/... -> ./node_modules/...
    } 
    else {
        filePath = '.' + urlPath;
    }

    console.log(`[SERVER] Serving file: ${filePath}`);

    const extname = path.extname(filePath);
    const contentType = MIME_TYPES[extname] || 'application/octet-stream';

    fs.readFile(filePath, (error, content) => {
        if (error) {
            console.error(`[SERVER] Error reading file: ${error.code} - ${filePath}`);
            if (error.code === 'ENOENT') {
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end('404: File non trovato', 'utf-8');
            } else if (error.code === 'EACCES') {
                res.writeHead(403, { 'Content-Type': 'text/plain' });
                res.end('403: Permesso negato', 'utf-8');
            } else {
                res.writeHead(500, { 'Content-Type': 'text/plain' });
                res.end(`Errore del server: ${error.code}`, 'utf-8');
            }
            return;
        }

        const headers = {
            'Content-Type': contentType,
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp',
            'Cross-Origin-Resource-Policy': 'cross-origin',
            'Access-Control-Allow-Origin': '*',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        };

        if (extname === '.gguf') {
            headers['Accept-Ranges'] = 'bytes';
        }

        res.writeHead(200, headers);
        res.end(content, 'utf-8');
    });
});

server.listen(PORT, '127.0.0.1', () => {
    console.log('\x1b[36m%s\x1b[0m', `[OUROBOROS KERNEL] Attivo su http://127.0.0.1:${PORT}`);
    console.log('\x1b[32m%s\x1b[0m', '[A.S.T.S.] Header di sicurezza COOP/COEP/Resource-Policy iniettati.');
    console.log('\x1b[33m%s\x1b[0m', `[INFO] Servire anche node_modules dalla root.`);
});

server.on('error', (err) => {
    console.error('\x1b[31m%s\x1b[0m', `Errore del server: ${err.message}`);
    if (err.code === 'EADDRINUSE') {
        console.error('\x1b[31m%s\x1b[0m', `La porta ${PORT} è già in uso. Cambia la porta o arresta il processo in esecuzione.`);
    }
});

process.on('SIGINT', () => {
    console.log('\n\x1b[35m%s\x1b[0m', '[OUROBOROS] Server arrestato manualmente.');
    process.exit(0);
});