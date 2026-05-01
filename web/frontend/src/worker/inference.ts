import { loadModel } from "./model-cache";
import { computeMelSpectrogram } from "./mel";
import { CTCDecoder } from "./ctc-decode";
import { createSession, runInference } from "./session";
import { beamSearchDecode } from "./beam-decode";
import { buildTrie, type CompactTrie } from "../lib/phoneme-trie";
import { QuranDB } from "../lib/quran-db";
import { RecitationTracker } from "../lib/tracker";
import type { TranscribeResult, BeamVerseMatch } from "../lib/tracker";
import type { QuranVerse, WorkerInbound, WorkerOutbound } from "../lib/types";

const MODEL_URL = "/fastconformer_phoneme_q8.onnx";

let tracker: RecitationTracker | null = null;
let decoder: CTCDecoder | null = null;
let db: QuranDB | null = null;
let trie: CompactTrie | null = null;
let vocabJsonCache: Record<string, string> | null = null;
let quranDataCache: QuranVerse[] | null = null;
let debugEnabled = false;

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

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const numMels = 80;
  const { logprobs, timeSteps, vocabSize } = await runInference(
    features,
    numMels,
    timeFrames,
  );

  const greedy = decoder!.decode(logprobs, timeSteps, vocabSize);

  // Run trie-constrained beam search for verse-level matches
  let beamMatches: BeamVerseMatch[] | undefined;
  const beamDebug: Array<{
    ref: string;
    spanLength: number;
    score: number;
  }> = [];
  if (trie) {
    const beamResults = beamSearchDecode(
      logprobs, timeSteps, vocabSize,
      decoder!.getBlankId(), trie, 8,
    );
    // Collect verse matches from beam hypotheses
    const seen = new Set<string>();
    beamMatches = [];
    for (const result of beamResults) {
      for (const ref of result.matchedVerses) {
        const key = `${ref.verseIndex}:${ref.spanLength}`;
        if (!seen.has(key)) {
          seen.add(key);
          beamMatches.push({
            verseIndex: ref.verseIndex,
            spanLength: ref.spanLength,
            score: result.score,
          });
          if (debugEnabled) {
            const verse = quranDataCache?.[ref.verseIndex];
            beamDebug.push({
              ref: verse ? `${verse.surah}:${verse.ayah}` : `idx:${ref.verseIndex}`,
              spanLength: ref.spanLength,
              score: Math.round(result.score * 1000) / 1000,
            });
          }
        }
      }
    }
  }

  postDebug("transcribe", {
    audioSec: Math.round((audio.length / 16000) * 100) / 100,
    text: greedy.text,
    rawPhonemes: greedy.rawPhonemes,
    tokenCount: greedy.tokenIds?.length ?? 0,
    beam: beamDebug.slice(0, 8),
  });

  return {
    ...greedy,
    acoustic: {
      logprobs,
      timeSteps,
      vocabSize,
      blankId: decoder!.getBlankId(),
    },
    beamMatches,
  };
}

async function init() {
  try {
    // Load vocab
    post({ type: "loading_status", message: "Loading vocabulary..." });
    const vocabRes = await fetch("/phoneme_vocab.json");
    if (!vocabRes.ok) throw new Error(`phoneme_vocab.json fetch failed: ${vocabRes.status}`);
    const vocabJson = await vocabRes.json();
    vocabJsonCache = vocabJson;
    decoder = new CTCDecoder(vocabJson);

    // Load ONNX model
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

    // Load QuranDB (phoneme data)
    post({ type: "loading_status", message: "Loading Quran data..." });
    const quranRes = await fetch("/quran_phonemes.json");
    if (!quranRes.ok) throw new Error(`quran_phonemes.json fetch failed: ${quranRes.status}`);
    const quranData = await quranRes.json();
    quranDataCache = quranData;
    db = new QuranDB(quranData, decoder);

    // Build verse/span trie for constrained beam search
    post({ type: "loading_status", message: "Building search trie..." });
    const built = buildTrie(quranData, vocabJsonCache!, 3);
    trie = built.trie;
    console.log(
      `Trie built: ${built.stats.nodeCount} nodes, ` +
      `${built.stats.singleVerseCount} verses, ${built.stats.spanCount} spans, ` +
      `~${built.stats.memoryMB.toFixed(1)}MB`,
    );

    // Create tracker
    tracker = new RecitationTracker(db, transcribe, {
      onDiagnostic: (event) => postDebug("tracker", { ...event }),
    });
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
    if (db) {
      tracker = new RecitationTracker(db, transcribe, {
        onDiagnostic: (event) => postDebug("tracker", { ...event }),
      });
    }
  } else if (msg.type === "set_debug") {
    debugEnabled = msg.enabled;
  } else if (msg.type === "audio") {
    if (!tracker) return;
    const messages = await tracker.feed(msg.samples);
    for (const m of messages) {
      post(m);
    }
  }
};
