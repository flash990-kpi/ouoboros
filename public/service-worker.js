// Minimal service worker for offline caching of static assets
const CACHE = 'ouro-static-v1';
const ASSETS = [
  '/',
  '/index.html',
  '/public/manifest.json'
];
self.addEventListener('install', (ev) => {
  ev.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)));
  self.skipWaiting();
});
self.addEventListener('activate', (ev) => {
  ev.waitUntil(self.clients.claim());
});
self.addEventListener('fetch', (ev) => {
  const url = new URL(ev.request.url);
  if (url.origin !== location.origin) return; // only same-origin
  ev.respondWith(caches.match(ev.request).then((r) => r || fetch(ev.request)));
});
