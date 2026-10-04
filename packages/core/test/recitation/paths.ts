/**
 * Asset lookup for the recitation-engine tests.
 *
 * The oracle vectors are committed (`lab/docs/specs/vectors/`); the phoneme
 * corpus, the ONNX model and the audio clips are not, and a worktree under
 * `.worktrees/<name>/` has neither `data/` nor `web/frontend/node_modules`.
 * So every lookup walks the worktree first, then the main checkout.
 */
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

/** Repo root of the checkout this file lives in (worktree or main). */
export const REPO_ROOT = resolve(HERE, "../../../..");
/** Where `.worktrees/<name>` was branched from. Same as REPO_ROOT otherwise. */
export const MAIN_CHECKOUT = resolve(REPO_ROOT, "../..");

export const VECTORS = resolve(REPO_ROOT, "lab/docs/specs/vectors");

export function firstExisting(...paths: Array<string | undefined>): string | null {
  for (const p of paths) {
    if (p && existsSync(p)) return p;
  }
  return null;
}

/** Same path under the worktree and under the checkout it came from. */
function bothRoots(relative: string): string[] {
  return [resolve(REPO_ROOT, relative), resolve(MAIN_CHECKOUT, relative)];
}

export function findCorpus(): string | null {
  return firstExisting(
    process.env.ZIPFORMER_CORPUS,
    ...bothRoots("web/frontend/public/zipformer_quran.json"),
    ...bothRoots("lab/data/zipformer/zipformer_quran.json"),
  );
}

/** The default model (a0w-ep1-a0.5 int8). */
export function findModel(): string | null {
  return firstExisting(
    process.env.ZIPFORMER_MODEL,
    ...bothRoots("web/frontend/public/models/zipformer_a0w_ep1_a05.int8.onnx"),
  );
}

export function findModelIo(): string | null {
  return firstExisting(
    process.env.ZIPFORMER_IO,
    ...bothRoots("web/frontend/public/models/zipformer_a0w_ep1_a05.io.json"),
  );
}

/** The previous default (interp-gentle-a0.5 int8); the recorded CTC vectors come from it. */
export function findLegacyModel(): string | null {
  return firstExisting(
    process.env.ZIPFORMER_LEGACY_MODEL,
    ...bothRoots("web/frontend/public/models/zipformer_interp_gentle_a05.int8.onnx"),
    ...bothRoots("lab/data/zipformer/zipformer_interp_gentle_a05.int8.onnx"),
  );
}

export function findLegacyModelIo(): string | null {
  return firstExisting(
    process.env.ZIPFORMER_LEGACY_IO,
    ...bothRoots("web/frontend/public/models/zipformer_interp_gentle_a05.io.json"),
  );
}

/** `onnxruntime-node` lives in the web demo's `node_modules`. */
export function findOrtDir(): string | null {
  return firstExisting(
    process.env.ZIPFORMER_ORT_DIR,
    ...bothRoots("web/frontend/node_modules"),
  );
}

export function findPython(): string | null {
  return firstExisting(process.env.TILAWA_PY, ...bothRoots(".venv/bin/python"));
}

/** Root to put on `sys.path` for `from shared.audio import load_audio`. */
export function findPythonRoot(): string | null {
  const shared = firstExisting(...bothRoots("lab/shared/audio.py"));
  return shared ? resolve(dirname(shared), "..") : null;
}

export function findClip(name = "001002.mp3"): string | null {
  return firstExisting(
    process.env.TILAWA_CLIP,
    ...bothRoots(`lab/benchmark/test_corpus/${name}`),
  );
}

export function requireCorpus(): string {
  const corpus = findCorpus();
  if (!corpus) {
    throw new Error(
      "zipformer_quran.json not found. Set ZIPFORMER_CORPUS, or run " +
        "`bash web/frontend/scripts/fetch-zipformer-assets.sh`.",
    );
  }
  return corpus;
}
