import { IDBFactory } from "fake-indexeddb";
import { expect, it } from "vitest";
import { cleanupOldClient } from "../src/migrate";

function open(idb: IDBFactory, name: string, store: string, value?: unknown): Promise<void> {
  return new Promise((resolve, reject) => {
    const req = idb.open(name, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(store);
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const db = req.result;
      if (value === undefined) return db.close(), resolve();
      const tx = db.transaction(store, "readwrite");
      tx.objectStore(store).put(value, "k");
      tx.oncomplete = () => (db.close(), resolve());
      tx.onerror = () => reject(tx.error);
    };
  });
}

function fakeCaches(names: string[]) {
  const live = new Set(names);
  return {
    live,
    keys: async () => [...live],
    delete: async (k: string) => live.delete(k),
  };
}

it("deletes the 0.x model database and old caches, and keeps the SDK cache", async () => {
  const idb = new IDBFactory();
  await open(idb, "tarteel-models", "models", new Uint8Array(1024));
  await open(idb, "someone-else", "x");
  const caches = fakeCaches(["tilawa-models", "tilawa-zipformer-default-v2", "workbox-precache-v2"]);
  let unregistered = 0;
  const serviceWorker = {
    getRegistrations: async () => [{ unregister: async () => (unregistered++, true) }],
  } as unknown as ServiceWorkerContainer;

  const result = await cleanupOldClient({ indexedDB: idb, caches, serviceWorker });

  expect(result).toEqual({
    databases: ["tarteel-models"],
    caches: expect.arrayContaining(["tilawa-zipformer-default-v2", "workbox-precache-v2"]),
    serviceWorkers: 1,
  });
  expect([...caches.live]).toEqual(["tilawa-models"]);
  expect(unregistered).toBe(1);
  const names = (await idb.databases()).map((d) => d.name);
  expect(names).toEqual(["someone-else"]);
});

it("is a no-op on a fresh or already-cleaned browser", async () => {
  const idb = new IDBFactory();
  const caches = fakeCaches(["tilawa-models"]);
  const first = await cleanupOldClient({ indexedDB: idb, caches });
  const second = await cleanupOldClient({ indexedDB: idb, caches });
  expect(first.caches).toEqual([]);
  expect(second).toEqual(first);
  expect([...caches.live]).toEqual(["tilawa-models"]);
  expect(await idb.databases()).toEqual([]);
});

it("survives browsers without the APIs", async () => {
  expect(await cleanupOldClient({})).toEqual({ databases: [], caches: [], serviceWorkers: 0 });
});
