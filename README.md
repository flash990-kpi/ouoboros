# Ouroboros - README

Questo branch contiene una implementazione runtime iniziale per A.S.T.S. (Adaptive Sparse Topology Synthesis).

Prerequisiti:
- Node.js 18+ (consigliato)

Come avviare (sviluppo rapido):

1. npm install
2. npm start

Questo avvierà il server integrato che serve i file statici e supporta richieste Range per file .gguf.

Importante:
- Per utilizzare SharedArrayBuffer in browser, il server deve rispondere con header COOP/COEP: same-origin / require-corp.
- GitHub Pages non permette COOP/COEP necessari; usa VPS o Cloud che consenta impostare header.

Contenuto rilevante:
- server.js: server statico con supporto Range e header di sicurezza
- hw/: driver e auditor
- asts/: topology parser e weight synthesizer (TS/JS)
- io/: gguf streamer
- public/: manifest e service worker

Prossimi passi:
- Eseguire build dei moduli front-end (se necessario usare esbuild/rollup/tsc)
- Integrare kernel WGSL per WebGPU e WASM ottimizzato

