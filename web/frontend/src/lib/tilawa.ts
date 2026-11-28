import type { LoadProgress, Tilawa } from "@tilawa/core/web";
import { indexQuran, type QuranText, type QuranVerse } from "./quran-words";

/** Same-origin mirror of https://tilawa.dev/v1/ (served by server/assets.ts from dist/v1). */
export const ASSET_BASE = new URL(import.meta.env.VITE_TILAWA_ASSET_BASE ?? "/v1/", location.href).href;

const params = new URLSearchParams(location.search);
// Not in the UI: lets tests and developers pin a tier (?tier=lite).
export const TIER: "auto" | "lite" | "full" =
  params.get("tier") === "lite" || params.get("tier") === "full" ? (params.get("tier") as "lite" | "full") : "auto";

const STAGE_WEIGHT = { model: 0.75, reference: 0.2, warmup: 0.05 } as const;

export interface TilawaLoadUpdate {
  overall: number;
  pct: number;
  cached: boolean;
  stage: LoadProgress["stage"];
}

export interface LoadTilawaOptions {
  signal?: AbortSignal;
  onProgress?: (update: TilawaLoadUpdate) => void;
  onOrtVersion?: (version: string) => void;
}

let quranPromise: Promise<QuranText> | null = null;
export function loadQuran(): Promise<QuranText> {
  quranPromise ??= fetch("/quran.json")
    .then((r) => {
      if (!r.ok) throw new Error(`${r.status} fetching the Quran text`);
      return r.json() as Promise<QuranVerse[]>;
    })
    .then(indexQuran);
  quranPromise.catch(() => (quranPromise = null));
  return quranPromise;
}

let tilawaPromise: Promise<Tilawa> | null = null;
export function loadTilawa(options: LoadTilawaOptions = {}): Promise<Tilawa> {
  if (tilawaPromise) return tilawaPromise;
  const done = { model: 0, reference: 0, warmup: 0 };
  let cached = true;
  tilawaPromise = Promise.all([import("@tilawa/core/web"), import("onnxruntime-web/all")]).then(([web, ort]) => {
    options.onOrtVersion?.(ort.env.versions?.web ?? "");
    return web.createTilawa({
      assetBase: ASSET_BASE,
      tier: TIER,
      signal: options.signal,
      onProgress: (p: LoadProgress) => {
        done[p.stage] = p.fraction;
        if (!p.fromCache && p.stage !== "warmup" && p.fraction < 1) cached = false;
        const overall = Object.entries(done).reduce((n, [k, f]) => n + f * STAGE_WEIGHT[k as keyof typeof done], 0);
        options.onProgress?.({
          overall,
          pct: Math.floor(overall * 100),
          cached,
          stage: p.stage,
        });
      },
    });
  });
  tilawaPromise.catch(() => (tilawaPromise = null));
  return tilawaPromise;
}
