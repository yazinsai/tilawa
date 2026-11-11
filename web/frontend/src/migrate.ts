/**
 * First-load cleanup for browsers that ran the 0.x demo. The old demo kept its ~66 MB model in the IndexedDB
 * database `tarteel-models` and precached files with a service worker; V1 keeps model and reference bytes in the
 * Cache API cache `tilawa-models` (the SDK's), so everything else is orphaned. Idempotent: safe on every load.
 */

export const OLD_DATABASES = ["tarteel-models"];
export const KEPT_CACHES = ["tilawa-models"];

export interface CleanupEnv {
  indexedDB?: Pick<IDBFactory, "deleteDatabase">;
  caches?: Pick<CacheStorage, "keys" | "delete">;
  serviceWorker?: Pick<ServiceWorkerContainer, "getRegistrations">;
}

export interface CleanupResult {
  databases: string[];
  caches: string[];
  serviceWorkers: number;
}

function deleteDatabase(idb: CleanupEnv["indexedDB"] & object, name: string): Promise<boolean> {
  return new Promise((resolve) => {
    let req: IDBOpenDBRequest;
    try {
      req = idb.deleteDatabase(name);
    } catch {
      return resolve(false);
    }
    req.onsuccess = () => resolve(true);
    req.onerror = () => resolve(false);
    // Another tab still has it open: the delete finishes when that tab closes it.
    req.onblocked = () => resolve(true);
  });
}

export async function cleanupOldClient(env: CleanupEnv = browserEnv()): Promise<CleanupResult> {
  const result: CleanupResult = { databases: [], caches: [], serviceWorkers: 0 };
  const jobs: Promise<unknown>[] = [];
  if (env.serviceWorker) {
    jobs.push(
      env.serviceWorker
        .getRegistrations()
        .then((regs) => Promise.all(regs.map((r) => r.unregister().then((ok) => ok && result.serviceWorkers++))))
        .catch(() => {}),
    );
  }
  if (env.indexedDB) {
    const idb = env.indexedDB;
    for (const name of OLD_DATABASES)
      jobs.push(deleteDatabase(idb, name).then((ok) => ok && result.databases.push(name)));
  }
  if (env.caches) {
    const cs = env.caches;
    jobs.push(
      cs
        .keys()
        .then((keys) =>
          Promise.all(
            keys
              .filter((k) => !KEPT_CACHES.includes(k))
              .map((k) => cs.delete(k).then((ok) => ok && result.caches.push(k))),
          ),
        )
        .catch(() => {}),
    );
  }
  await Promise.all(jobs);
  return result;
}

function browserEnv(): CleanupEnv {
  const g = globalThis as typeof globalThis & { indexedDB?: IDBFactory; caches?: CacheStorage };
  return {
    indexedDB: g.indexedDB,
    caches: g.caches,
    serviceWorker: typeof navigator !== "undefined" && "serviceWorker" in navigator ? navigator.serviceWorker : undefined,
  };
}
