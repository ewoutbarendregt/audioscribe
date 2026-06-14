// Service Worker for Audioscribe PWA
const CACHE_NAME = 'audioscribe-v3';
// Relative so they resolve under the app's mount path (e.g. /projects/audioscribe/)
const STATIC_ASSETS = [
    './',
    './manifest.json'
];

// Install: cache static assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(STATIC_ASSETS);
        })
    );
    self.skipWaiting();
});

// Activate: clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames
                    .filter((name) => name !== CACHE_NAME)
                    .map((name) => caches.delete(name))
            );
        })
    );
    self.clients.claim();
});

// Fetch: network-first for API, cache-first for static
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Always go to network for API calls (matches both root and prefixed mounts)
    if (url.pathname.includes('/api/')) {
        return;
    }

    // Network-first for navigations / the HTML document, so a freshly deployed
    // app is picked up on the next load (not one reload later). Falls back to
    // cache when offline.
    if (event.request.mode === 'navigate' || event.request.destination === 'document') {
        event.respondWith(
            fetch(event.request).then((response) => {
                if (response.ok) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
                }
                return response;
            }).catch(() => caches.match(event.request).then((c) => c || caches.match('./')))
        );
        return;
    }

    // Cache-first for other static assets
    event.respondWith(
        caches.match(event.request).then((cached) => {
            if (cached) {
                // Return cached, but also update cache in background
                fetch(event.request).then((response) => {
                    if (response.ok) {
                        caches.open(CACHE_NAME).then((cache) => {
                            cache.put(event.request, response);
                        });
                    }
                }).catch(() => {});
                return cached;
            }

            // Not in cache, fetch from network
            return fetch(event.request).then((response) => {
                if (response.ok && event.request.method === 'GET') {
                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, responseClone);
                    });
                }
                return response;
            });
        })
    );
});
