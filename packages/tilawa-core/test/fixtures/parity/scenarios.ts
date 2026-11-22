/**
 * Parity scenarios, run against the research session (scripts/parity-golden.mjs, at the pinned commit) and the public
 * session (test/parity.test.ts). Only plain types cross this module, so both builds can import it.
 */
export interface WordRef { surah: number; ayah: number; word: number }
interface AnySession {
  subscribe(fn: (e: unknown) => void): () => void;
  pushAudio(samples: Float32Array, rate: number): Promise<unknown[]>;
  end(): Promise<unknown[]>;
  pushEvidence(e: unknown): unknown[];
  endEvidence(): unknown[];
  mistakes(): unknown[];
  evidence(): unknown[];
}
export interface ParityApi {
  start(passage: { surah: number; fromAyah: number; toAyah: number } | null, policy?: Record<string, unknown>): AnySession;
  wav: { samples: Float32Array; sampleRate: number };
  /** syntheticFixture(index, spec).evidence */
  synthetic(spec: { id: string; from: WordRef; to: WordRef; edits: unknown[]; chunkSize: number }): unknown[];
  index: { words: { surah: number; ayah: number; word: number; start: number; end: number }[]; phonemes: string };
}
/** The engine must be created with `recordEvidence: true`, so `evidence` covers the listener's output too. */
export interface ScenarioResult { returned: unknown[]; subscribed: unknown[]; mistakes: unknown[]; evidence: unknown[] }

const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });

async function audio(api: ParityApi, passage: Parameters<ParityApi["start"]>[0], chunk: number, resampleTo?: number): Promise<ScenarioResult> {
  const s = api.start(passage);
  const subscribed: unknown[] = [];
  s.subscribe((e) => subscribed.push(e));
  let { samples, sampleRate } = api.wav;
  if (resampleTo) {
    // Nearest-sample upsampling: a deterministic non-16 kHz input for the listener's resampler.
    const up = new Float32Array(Math.floor((samples.length * resampleTo) / sampleRate));
    for (let i = 0; i < up.length; i++) up[i] = samples[Math.min(samples.length - 1, Math.round((i * sampleRate) / resampleTo))];
    samples = up;
    sampleRate = resampleTo;
  }
  const pending: Promise<unknown[]>[] = [];
  // Not awaited between pushes: the session serialises them.
  for (let a = 0; a < samples.length; a += chunk) pending.push(s.pushAudio(samples.subarray(a, a + chunk), sampleRate));
  pending.push(s.end());
  const returned = (await Promise.all(pending)).flat();
  return { returned, subscribed, mistakes: s.mistakes(), evidence: s.evidence() };
}

function evidence(api: ParityApi, passage: Parameters<ParityApi["start"]>[0], policy: Record<string, unknown>, edits: (at: (r: WordRef) => number, text: (r: WordRef) => string) => unknown[], chunkSize: number): ScenarioResult {
  const find = (r: WordRef) => api.index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
  const from = w(2, 255, 1);
  const at = (r: WordRef) => find(r).start - find(from).start;
  const text = (r: WordRef) => api.index.phonemes.slice(find(r).start, find(r).end);
  const s = api.start(passage, policy);
  const subscribed: unknown[] = [];
  s.subscribe((e) => subscribed.push(e));
  const returned: unknown[] = [];
  for (const c of api.synthetic({ id: "parity", from, to: w(2, 256, 10), edits: edits(at, text), chunkSize })) returned.push(...s.pushEvidence(c));
  returned.push(...s.endEvidence());
  return { returned, subscribed, mistakes: s.mistakes(), evidence: s.evidence() };
}

const target = w(2, 255, 23);
const swap = (t: string) => t.replace("ش", "س").replace("ف", "م");

export async function runScenarios(api: ParityApi): Promise<Record<string, ScenarioResult>> {
  const passage = { surah: 2, fromAyah: 255, toAyah: 256 };
  return {
    "wav-free-8000": await audio(api, null, 8000),
    "wav-memorization-3200": await audio(api, passage, 3200),
    "wav-free-44100": await audio(api, null, 4410, 44100),
    "evidence-self-correction": evidence(api, passage, { evidenceMs: 60 },
      (at, text) => [{ kind: "ins", at: at(target), text: swap(text(target)) }], 2),
    "evidence-wrong-word-free": evidence(api, null, {},
      (at, text) => [{ kind: "del", at: at(target), count: text(target).length }, { kind: "ins", at: at(target), text: swap(text(target)) }], 3),
    "evidence-wrong-word-next-word": evidence(api, passage, { confirm: { after: "next-word" }, countSelfCorrections: true },
      (at, text) => [{ kind: "del", at: at(target), count: text(target).length }, { kind: "ins", at: at(target), text: swap(text(target)) }], 5),
  };
}
