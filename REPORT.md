# REPORT

Overview tecnico-commerciale sintetico.

1) Tecnica
La tecnica A.S.T.S. sfrutta un indice topologico (.ouro) per mappare quali cluster di pesi sono necessari per una specifica inferenza e streaming chirurgico dal file GGUF. I pesi vengono rigenerati a rango ridotto (low-rank) e inviati al driver hardware mediante micro-buffers.

2) Stato attuale in questo branch
- Server statico con Range support
- Driver manager per rilevamento WebNN/WebGPU/WASM
- Implementazioni JS/TS per parser topology e sintetizzatore peso
- PWA manifest + service worker

3) Metriche e stime (ordine di grandezza)
- Un device mobile top-tier con NPU/WebNN ottimizzato: 5-20 token/s (stima ottimistica) su 70B con grande riduzione di peso attivo.
- Desktop GPU: 5-50 token/s a seconda di ottimizzazione WGSL.

4) Valore IP
Range stimato: da qualche milione a qualche decina di milioni USD; dipende dai benchmark e protezione IP (brevettabilità).

5) Limitazioni
- WebNN è ancora in evoluzione ed implementazioni variano tra browser
- La qualità dell'approssimazione low-rank dipende fortemente dal tipo di tensore

6) Azioni raccomandate
- Benchmark su 3 dispositivi reali
- Sviluppo WGSL kernels per MatMul/LayerNorm/Softmax
- Compilare WASM con SIMD per fallback CPU

