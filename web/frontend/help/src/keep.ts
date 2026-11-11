// Opt-in "keep on this device for 7 days": the encoded report (json + wav) in IndexedDB, so a failed send can be
// retried after a refresh. Off by default; expired entries are deleted whenever /help opens.
const DB = "tilawa-help";
const STORE = "unsent";
export const KEEP_MS = 7 * 86_400_000;

export interface KeptTake {
  reportId: string;
  json: string;
  wav: ArrayBuffer;
  savedAt: number;
  expiresAt: number;
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in globalThis)) return reject(new Error("IndexedDB unavailable"));
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "reportId" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(mode: IDBTransactionMode, run: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  try {
    return await new Promise<T>((resolve, reject) => {
      const t = db.transaction(STORE, mode);
      const req = run(t.objectStore(STORE));
      t.oncomplete = () => resolve(req.result);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error);
    });
  } finally {
    db.close();
  }
}

export async function keepTake(reportId: string, json: string, wav: Uint8Array, now = Date.now()): Promise<void> {
  const copy = wav.slice().buffer as ArrayBuffer;
  const row: KeptTake = { reportId, json, wav: copy, savedAt: now, expiresAt: now + KEEP_MS };
  await tx("readwrite", (s) => s.put(row));
}

export async function dropTake(reportId: string): Promise<void> {
  await tx("readwrite", (s) => s.delete(reportId));
}

/** Unexpired kept takes, oldest first; expired ones are deleted on the way. */
export async function keptTakes(now = Date.now()): Promise<KeptTake[]> {
  const all = (await tx("readonly", (s) => s.getAll())) as KeptTake[];
  const live: KeptTake[] = [];
  for (const row of all) {
    if (row.expiresAt <= now) await dropTake(row.reportId).catch(() => undefined);
    else live.push(row);
  }
  return live.sort((a, b) => a.savedAt - b.savedAt);
}
