import { QuranDB } from "../lib/quran-db";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";
import { RecitationTracker } from "../lib/tracker";
import type { TranscribeResult } from "../lib/tracker";
import { loadModel } from "./model-cache";
import {
  adaptQuranTextData,
  validateCtcTokenRoundTrip,
  type CtcTokenTable,
} from "./quran-text-adapter";
import { DEFAULT_STREAMING_CONFIG, normalizeStreamingConfig, type StreamingConfig } from "../lib/types";
import { createSession, runInference } from "./session";
import { TextCTCDecoder } from "./text-ctc-decode";

const MODEL_URL = "/fastconformer_full_mixed.onnx";
const VOCAB_URL = "/vocab.json";
const QURAN_URL = "/quran.json";
const CTC_TOKENS_URL = "/quran_ctc_tokens.json";
const METADATA_URL = "/export_metadata.json";

let tracker: RecitationTracker | null = null;
let decoder: TextCTCDecoder | null = null;
let db: QuranDB | null = null;
let debugEnabled = false;
let activeConfig: StreamingConfig = DEFAULT_STREAMING_CONFIG;

function post(msg: WorkerOutbound) {
  self.postMessage(msg);
}

function postDebug(event: string, data: Record<string, unknown>) {
  if (!debugEnabled) return;
  post({
    type: "debug",
    event,
    at: Date.now(),
    data,
  });
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

function makeTracker(): RecitationTracker {
  if (!db) throw new Error("QuranDB not initialized");
  return new RecitationTracker(db, transcribe, {
    config: activeConfig,
    onDiagnostic: (event) => postDebug("tracker", { ...event }),
  });
}

function setConfig(config: Partial<StreamingConfig>): void {
  activeConfig = normalizeStreamingConfig(config);
  tracker?.setConfig(activeConfig);
  postDebug("config", activeConfig as unknown as Record<string, unknown>);
}

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  if (!decoder || !db) throw new Error("Worker not initialized");

  const { logprobs, timeSteps, vocabSize } = await runInference(audio);
  const greedy = decoder.decode(logprobs, timeSteps, vocabSize);
  const champion = db.bestJoint03Match(greedy.text);

  postDebug("transcribe", {
    audioSec: Math.round((audio.length / 16000) * 100) / 100,
    text: greedy.text,
    tokenCount: greedy.tokenIds.length,
    champion: champion
      ? {
          ref: `${champion.surah}:${champion.ayah}` +
            (champion.ayah_end ? `-${champion.ayah_end}` : ""),
          score: champion.score,
        }
      : null,
  });

  const trustedChampion = champion?.score && champion.score >= 0.8 ? champion : null;
  return {
    text: greedy.text,
    rawPhonemes: greedy.text,
    tokenIds: greedy.tokenIds,
    acoustic: {
      logprobs,
      timeSteps,
      vocabSize,
      blankId: decoder.getBlankId(),
    },
    championMatch: trustedChampion ?? undefined,
  };
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
    decoder = new TextCTCDecoder(vocabJson, Number(metadata.blank_id ?? 1024));

    post({ type: "loading_status", message: "Loading Quran token table..." });
    const { json: ctcTokens } = await fetchJsonWithHash<CtcTokenTable>(CTC_TOKENS_URL);

    post({ type: "loading_status", message: "Loading Quran data..." });
    const { json: quranRaw } = await fetchJsonWithHash<unknown[]>(QURAN_URL);
    const quranData = adaptQuranTextData(quranRaw as any[], ctcTokens, decoder);
    const tokenErrors = validateCtcTokenRoundTrip(quranData, decoder);
    if (tokenErrors.length > 0) {
      console.warn("Quran CTC token round-trip warnings:", tokenErrors.slice(0, 8));
    }
    db = new QuranDB(quranData, undefined, ctcTokens);

    post({ type: "loading_status", message: "Downloading model..." });
    const modelBuffer = await loadModel(
      MODEL_URL,
      (loaded, total) => {
        post({
          type: "loading",
          percent: total ? Math.round((loaded / total) * 100) : 0,
        });
      },
    );

    post({ type: "loading_status", message: "Creating inference session..." });
    await createSession(modelBuffer);

    tracker = makeTracker();
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
    if (db) tracker = makeTracker();
  } else if (msg.type === "set_debug") {
    debugEnabled = msg.enabled;
  } else if (msg.type === "set_config") {
    setConfig(msg.config);
  } else if (msg.type === "audio") {
    if (!tracker) return;
    const messages = await tracker.feed(msg.samples);
    for (const m of messages) {
      post(m);
    }
  }
};
