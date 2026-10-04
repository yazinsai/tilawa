// Unregister old service worker and clear caches.
// Bump CACHE_VERSION whenever the default engine or model URLs change so
// existing clients replace this file and drop stale precaches.
const CACHE_VERSION = "tilawa-zipformer-default-v2";
const PRECACHE_URLS = [
  "/models/zipformer_a0w_ep1_a05.int8.onnx",
  "/models/zipformer_a0w_ep1_a05.io.json",
  "/zipformer_quran.json",
];

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", async () => {
  const keys = await caches.keys();
  await Promise.all(
    keys
      .filter((k) => k !== CACHE_VERSION)
      .map((k) => caches.delete(k)),
  );
  try {
    await caches.delete(CACHE_VERSION);
  } catch {
    /* ignore */
  }
  void PRECACHE_URLS;
  const clients = await self.clients.matchAll({ type: "window" });
  for (const client of clients) client.navigate(client.url);
  await self.registration.unregister();
});
