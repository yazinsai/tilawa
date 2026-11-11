// Kill switch for the 0.x demo's service worker: browsers that still have it registered fetch this file on their
// next update check. It drops the old precaches, reloads open tabs and unregisters itself. The page itself never
// registers a service worker any more (src/migrate.ts unregisters any it finds).
// `tilawa-models` is the V1 SDK's model cache: keep it.
const KEEP = ["tilawa-models"];

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.filter((k) => !KEEP.includes(k)).map((k) => caches.delete(k)));
      const clients = await self.clients.matchAll({ type: "window" });
      for (const client of clients) client.navigate(client.url);
      await self.registration.unregister();
    })(),
  );
});
