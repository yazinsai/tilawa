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
const JOINT03_BEAM_WIDTH = 6;
const JOINT03_TOP_SYMBOLS = 8;
const JOINT03_MAX_HYPOTHESES = 4;
const JOINT03_SECOND_PASS_MATCH_GATE = 0.63;
const JOINT03_SECOND_PASS_FRAME_MEAN_MAX_LOGP = -0.42;
const JOINT03_DECODE2_BEAM_WIDTH = 11;
const JOINT03_DECODE2_TOP_SYMBOLS = 12;
const JOINT03_DECODE2_MAX_EXTRA_STRINGS = 4;

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

function logAddExp(a: number | undefined, b: number): number {
  if (a === undefined) return b;
  const hi = Math.max(a, b);
  const lo = Math.min(a, b);
  return hi + Math.log1p(Math.exp(lo - hi));
}

function topSymbolsForFrame(
  logprobs: Float32Array,
  offset: number,
  vocabSize: number,
  topSymbols: number,
): number[] {
  const keep = Math.min(topSymbols, vocabSize);
  const out: Array<{ id: number; value: number }> = [];
  for (let id = 0; id < vocabSize; id++) {
    const value = logprobs[offset + id];
    if (out.length < keep) {
      out.push({ id, value });
      out.sort((a, b) => b.value - a.value);
    } else if (value > out[out.length - 1].value) {
      out[out.length - 1] = { id, value };
      out.sort((a, b) => b.value - a.value);
    }
  }
  return out.map((entry) => entry.id);
}

function ctcPrefixBeamDecode(
  logprobs: Float32Array,
  timeSteps: number,
  vocabSize: number,
  blankId: number,
  beamWidth: number,
  topSymbols: number,
): Array<{ ids: number[]; score: number }> {
  let beam = new Map<string, { ids: number[]; score: number }>();
  beam.set("", { ids: [], score: 0 });

  for (let t = 0; t < timeSteps; t++) {
    const offset = t * vocabSize;
    const symbols = topSymbolsForFrame(logprobs, offset, vocabSize, topSymbols);
    const nextBeam = new Map<string, { ids: number[]; score: number }>();
    const items = [...beam.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(beamWidth * 3, beamWidth));

    for (const item of items) {
      for (const c of symbols) {
        const logp = item.score + logprobs[offset + c];
        let ids = item.ids;
        if (c !== blankId && item.ids[item.ids.length - 1] !== c) {
          ids = [...item.ids, c];
        }
        const key = ids.join(",");
        nextBeam.set(key, {
          ids,
          score: logAddExp(nextBeam.get(key)?.score, logp),
        });
      }
    }

    beam = new Map(
      [...nextBeam.entries()]
        .sort((a, b) => b[1].score - a[1].score)
        .slice(0, beamWidth),
    );
  }

  return [...beam.values()].sort((a, b) => b.score - a.score).slice(0, beamWidth);
}

function appendBeamHypotheses(
  hypotheses: string[],
  seen: Set<string>,
  logprobs: Float32Array,
  timeSteps: number,
  vocabSize: number,
  blankId: number,
  beamWidth: number,
  topSymbols: number,
  maxExtra: number,
) {
  for (const result of ctcPrefixBeamDecode(
    logprobs,
    timeSteps,
    vocabSize,
    blankId,
    beamWidth,
    topSymbols,
  )) {
    const text = decoder!.tokenIdsToText(result.ids);
    if (!text.trim() || seen.has(text)) continue;
    seen.add(text);
    hypotheses.push(text);
    if (hypotheses.length >= maxExtra) break;
  }
}

function meanFrameMaxLogp(
  logprobs: Float32Array,
  timeSteps: number,
  vocabSize: number,
): number {
  let sum = 0;
  for (let t = 0; t < timeSteps; t++) {
    const offset = t * vocabSize;
    let max = logprobs[offset];
    for (let v = 1; v < vocabSize; v++) {
      max = Math.max(max, logprobs[offset + v]);
    }
    sum += max;
  }
  return sum / Math.max(1, timeSteps);
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
  const blankId = decoder!.getBlankId();
  const hypotheses: string[] = [];
  const seenHypotheses = new Set<string>();
  if (greedy.text.trim()) {
    hypotheses.push(greedy.text);
    seenHypotheses.add(greedy.text);
  }
  appendBeamHypotheses(
    hypotheses,
    seenHypotheses,
    logprobs,
    timeSteps,
    vocabSize,
    blankId,
    JOINT03_BEAM_WIDTH,
    JOINT03_TOP_SYMBOLS,
    JOINT03_MAX_HYPOTHESES,
  );
  const firstPass = db!.bestJoint03MatchForHypotheses(hypotheses);
  if (
    !firstPass ||
    firstPass.match.score < JOINT03_SECOND_PASS_MATCH_GATE ||
    meanFrameMaxLogp(logprobs, timeSteps, vocabSize) < JOINT03_SECOND_PASS_FRAME_MEAN_MAX_LOGP
  ) {
    const before = hypotheses.length;
    appendBeamHypotheses(
      hypotheses,
      seenHypotheses,
      logprobs,
      timeSteps,
      vocabSize,
      blankId,
      JOINT03_DECODE2_BEAM_WIDTH,
      JOINT03_DECODE2_TOP_SYMBOLS,
      before + JOINT03_DECODE2_MAX_EXTRA_STRINGS,
    );
  }
  const champion = db!.bestJoint03MatchForHypotheses(hypotheses);
  const championTranscript = champion?.transcript;

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
    text: championTranscript ?? greedy.text,
    rawPhonemes: greedy.rawPhonemes,
    tokenCount: greedy.tokenIds?.length ?? 0,
    beam: beamDebug.slice(0, 8),
    champion: champion
      ? {
          ref: `${champion.match.surah}:${champion.match.ayah}` +
            (champion.match.ayah_end ? `-${champion.match.ayah_end}` : ""),
          score: champion.match.score,
          transcript: champion.transcript,
        }
      : null,
  });

  return {
    ...greedy,
    text: championTranscript ?? greedy.text,
    acoustic: {
      logprobs,
      timeSteps,
      vocabSize,
      blankId,
    },
    beamMatches,
    championMatch: champion?.match,
    championTranscript,
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
