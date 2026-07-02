import {
  createTilawaSession,
  type CtcTokenTable,
  type TilawaSession,
} from "@tilawa/core";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";
import { loadModel } from "./model-cache";
import { createSession, runInference } from "./session";

const MODEL_URL = "/fastconformer_full_mixed.onnx";
const VOCAB_URL = "/vocab.json";
const QURAN_URL = "/quran.json";
const CTC_TOKENS_URL = "/quran_ctc_tokens.json";
const METADATA_URL = "/export_metadata.json";

let session: TilawaSession | null = null;
let debugEnabled = false;

function post(msg: WorkerOutbound) {
  self.postMessage(msg);
}

function postDebug(event: string, data: Record<string, unknown>) {
  if (!debugEnabled) return;
  post({ type: "debug", event, at: Date.now(), data });
}

async function sha256Hex(data: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function fetchJsonWithHash<T>(url: string): Promise<{ json: T; sha256: string }> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} fetch failed: ${res.status}`);
  const buffer = await res.arrayBuffer();
  const sha256 = await sha256Hex(buffer.slice(0));
  const json = JSON.parse(new TextDecoder().decode(buffer)) as T;
  return { json, sha256 };
}

async function init() {
  try {
    post({ type: "loading_status", message: "Loading metadata..." });
    const { json: metadata } = await fetchJsonWithHash<Record<string, unknown>>(METADATA_URL);

    post({ type: "loading_status", message: "Loading vocabulary..." });
    const { json: vocabJson, sha256: vocabSha256 } =
      await fetchJsonWithHash<Record<string, string>>(VOCAB_URL);
    const expectedVocabSha = metadata.vocab_sha256;
    if (typeof expectedVocabSha === "string" && expectedVocabSha !== vocabSha256) {
      throw new Error("vocab.json sha256 does not match export_metadata.json");
    }

    post({ type: "loading_status", message: "Loading Quran token table..." });
    const { json: ctcTokens } = await fetchJsonWithHash<CtcTokenTable>(CTC_TOKENS_URL);

    post({ type: "loading_status", message: "Loading Quran data..." });
    const { json: quranRaw } = await fetchJsonWithHash<unknown[]>(QURAN_URL);

    post({ type: "loading_status", message: "Downloading model..." });
    const modelBuffer = await loadModel(MODEL_URL, (loaded, total) => {
      post({
        type: "loading",
        percent: total ? Math.round((loaded / total) * 100) : 0,
      });
    });

    post({ type: "loading_status", message: "Creating inference session..." });
    await createSession(modelBuffer);

    session = createTilawaSession(
      { run: runInference },
      {
        vocab: vocabJson,
        quranCtcTokens: ctcTokens,
        quran: quranRaw,
        blankId: Number(metadata.blank_id ?? 1024),
      },
      {
        onOutput: (msg) => post(msg),
        onDiagnostic: (event, data) => postDebug(event, data),
      },
    );

    post({ type: "ready" });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("Worker init failed:", message);
    post({ type: "error", message });
  }
}

self.onmessage = async (e: MessageEvent<WorkerInbound>) => {
  const msg = e.data;
  if (msg.type === "init") {
    await init();
  } else if (msg.type === "reset") {
    session?.reset();
  } else if (msg.type === "set_debug") {
    debugEnabled = msg.enabled;
  } else if (msg.type === "set_config") {
    session?.setConfig(msg.config);
  } else if (msg.type === "audio") {
    if (!session) return;
    await session.feed(msg.samples);
  }
};
