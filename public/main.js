// Minimal glue for browser app: binds UI to driver manager and orchestrates simple flow
import { createDriverManager } from './hw/driverManager.js';

async function init() {
  const log = (m) => {
    const el = document.getElementById('file-status') || document.createElement('div');
    el.id = 'file-status';
    el.textContent = m;
    document.body.appendChild(el);
    console.log(m);
  };

  log('Eseguo audit hardware...');
  try {
    const { driver, profile } = await createDriverManager();
    log('Driver inizializzato: ' + (profile.hasWebNN ? 'WebNN' : profile.hasWebGPU ? 'WebGPU' : 'WASM/JS'));

    // Wire UI elements
    const fileInput = document.getElementById('file-input-hidden') || document.querySelector('input[type=file]');
    const executeBtn = document.getElementById('btn-execute');
    if (executeBtn) {
      executeBtn.disabled = false;
      executeBtn.addEventListener('click', async () => {
        log('Avvio procedura di inferenza (mock).');
        // Here we would orchestrate: read .ouro, build path, stream gguf chunks, synthesize weights, run driver.runInference
        try {
          // mock run
          await new Promise((r) => setTimeout(r, 800));
          log('Inferenza simulata completata.');
        } catch (e) {
          log('Errore inferenza: ' + e.message);
        }
      });
    }

  } catch (e) {
    console.error(e);
    const el = document.getElementById('file-status');
    if (el) el.textContent = 'Errore inizializzazione driver: ' + e.message;
  }

  // Register service worker
  if ('serviceWorker' in navigator) {
    try {
      await navigator.serviceWorker.register('/public/service-worker.js');
      log('Service worker registrato.');
    } catch (e) {
      log('Service worker error: ' + e.message);
    }
  }
}

window.addEventListener('load', init);
