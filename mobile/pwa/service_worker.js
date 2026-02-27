/**
 * Service Worker - BI-IDE v8 PWA
 * 
 * المميزات:
 * - Offline support (العمل بدون إنترنت)
 * - Background sync (المزامنة في الخلفية)
 * - Push notifications (الإشعارات)
 * - Cache strategies (استراتيجيات التخزين المؤقت)
 */

const CACHE_NAME = 'bi-ide-v8-cache-v1';
const STATIC_CACHE = 'bi-ide-static-v1';
const DYNAMIC_CACHE = 'bi-ide-dynamic-v1';
const API_CACHE = 'bi-ide-api-v1';

// Resources to pre-cache (الموارد للتخزين المسبق)
const PRECACHE_ASSETS = [
  '/',
  '/index.html',
  '/static/css/main.css',
  '/static/js/main.js',
  '/static/js/chunk-vendors.js',
  '/static/icons/icon-192x192.png',
  '/static/icons/icon-512x512.png',
  '/static/offline.html',
  // Fonts
  '/static/fonts/arabic-font.woff2',
  '/static/fonts/inter.woff2'
];

// API routes to cache (مسارات API للتخزين)
const API_ROUTES = [
  '/api/erp/dashboard',
  '/api/erp/inventory/summary',
  '/api/erp/accounting/reports',
  '/api/user/profile'
];

// Install event - Precache static assets
self.addEventListener('install', (event) => {
  console.log('[SW] Installing Service Worker...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[SW] Pre-caching static assets');
        return cache.addAll(PRECACHE_ASSETS);
      })
      .then(() => {
        console.log('[SW] Skip waiting');
        return self.skipWaiting();
      })
      .catch((err) => {
        console.error('[SW] Pre-caching failed:', err);
      })
  );
});

// Activate event - Clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating Service Worker...');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            // Delete old caches
            if (![STATIC_CACHE, DYNAMIC_CACHE, API_CACHE].includes(cacheName)) {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('[SW] Claiming clients');
        return self.clients.claim();
      })
  );
});

// Fetch event - Cache strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip chrome-extension requests
  if (url.protocol === 'chrome-extension:') {
    return;
  }
  
  // Strategy 1: Network First for API calls
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirstStrategy(request));
    return;
  }
  
  // Strategy 2: Cache First for static assets
  if (isStaticAsset(url.pathname)) {
    event.respondWith(cacheFirstStrategy(request));
    return;
  }
  
  // Strategy 3: Stale While Revalidate for everything else
  event.respondWith(staleWhileRevalidateStrategy(request));
});

/**
 * Cache Strategies
 */

// Network First - Try network, fall back to cache
async function networkFirstStrategy(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      // Update cache
      const cache = await caches.open(API_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('[SW] Network failed, serving from cache:', request.url);
    const cachedResponse = await caches.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline fallback for API
    return new Response(
      JSON.stringify({
        error: 'offline',
        message: 'You are offline. Please check your connection.',
        cached: false
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

// Cache First - Try cache, fall back to network
async function cacheFirstStrategy(request) {
  const cachedResponse = await caches.match(request);
  
  if (cachedResponse) {
    return cachedResponse;
  }
  
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('[SW] Both cache and network failed:', request.url);
    
    // Return offline page for HTML requests
    if (request.headers.get('accept').includes('text/html')) {
      return caches.match('/static/offline.html');
    }
    
    throw error;
  }
}

// Stale While Revalidate - Serve from cache, update in background
async function staleWhileRevalidateStrategy(request) {
  const cachedResponse = await caches.match(request);
  
  const fetchPromise = fetch(request)
    .then((networkResponse) => {
      if (networkResponse.ok) {
        const cache = await caches.open(DYNAMIC_CACHE);
        cache.put(request, networkResponse.clone());
      }
      return networkResponse;
    })
    .catch((error) => {
      console.log('[SW] Background fetch failed:', error);
    });
  
  return cachedResponse || fetchPromise;
}

/**
 * Helper Functions
 */

function isStaticAsset(pathname) {
  const staticExtensions = [
    '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg',
    '.woff', '.woff2', '.ttf', '.eot', '.ico'
  ];
  return staticExtensions.some(ext => pathname.endsWith(ext));
}

/**
 * Background Sync (المزامنة في الخلفية)
 */
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-forms') {
    event.waitUntil(syncFormSubmissions());
  } else if (event.tag === 'sync-transactions') {
    event.waitUntil(syncTransactions());
  }
});

async function syncFormSubmissions() {
  // Get pending submissions from IndexedDB
  // and send them to the server
  console.log('[SW] Syncing form submissions...');
}

async function syncTransactions() {
  // Sync pending ERP transactions
  console.log('[SW] Syncing transactions...');
}

/**
 * Push Notifications (الإشعارات)
 */
self.addEventListener('push', (event) => {
  if (!event.data) return;
  
  const data = event.data.json();
  const options = {
    body: data.body || 'New notification',
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/badge-72x72.png',
    tag: data.tag || 'default',
    requireInteraction: data.requireInteraction || false,
    actions: data.actions || [],
    data: data.payload || {}
  };
  
  event.waitUntil(
    self.registration.showNotification(
      data.title || 'BI-IDE v8',
      options
    )
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  const notificationData = event.notification.data;
  let url = '/';
  
  if (notificationData && notificationData.url) {
    url = notificationData.url;
  }
  
  event.waitUntil(
    clients.matchAll({ type: 'window' })
      .then((clientList) => {
        // Focus existing window if open
        for (const client of clientList) {
          if (client.url === url && 'focus' in client) {
            return client.focus();
          }
        }
        // Open new window
        if (clients.openWindow) {
          return clients.openWindow(url);
        }
      })
  );
});

/**
 * Message Handler (معالجة الرسائل من التطبيق)
 */
self.addEventListener('message', (event) => {
  if (event.data && event.data.type) {
    switch (event.data.type) {
      case 'SKIP_WAITING':
        self.skipWaiting();
        break;
        
      case 'GET_VERSION':
        event.ports[0].postMessage({ version: 'v8.0.0' });
        break;
        
      case 'CLEAR_CACHE':
        event.waitUntil(
          caches.keys()
            .then((cacheNames) => {
              return Promise.all(
                cacheNames.map((cacheName) => caches.delete(cacheName))
              );
            })
            .then(() => {
              event.ports[0].postMessage({ success: true });
            })
        );
        break;
        
      default:
        console.log('[SW] Unknown message type:', event.data.type);
    }
  }
});

console.log('[SW] Service Worker loaded');
