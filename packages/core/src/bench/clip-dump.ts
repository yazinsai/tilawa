/**
 * Per-clip outcomes behind the scoreboard gates, for the dashboard's mistake review: every miss, false warning and late
 * tracker lock, with the written words, what the listener decoded for each, and where to seek in the audio.
 */
import type { CheckerEvent, WordRef } from "../events.js";
import type { ReferenceAsset, ReferenceIndex } from "../reference.js";
import { alignRef, attribute } from "./acceptance.js";
import type { AcceptanceClip, ClipScore, LabelEvent, StartGate } from "./acceptance.js";
import type { Segment } from "./jumps.js";

export const CLIP_DUMP_VERSION = 2;

export type DumpArm = "free" | "declared" | "oracle" | "leadFree" | "leadDeclared";

export interface DumpEvent {
  /** Original final checker event; present on reports, never on benchmark misses. */
  checkerEvent?: CheckerEvent;
  errorClass: string;
  /** "surah:ayah:word". */
  at: string;
  state: string;
  expected?: string;
  observed?: string;
  confidence?: number;
  /** Seconds into `audio` to start playback, about a word before the event. */
  seekS: number;
  /** Audio file under the dashboard's /audio/ route; the clip's own unless the event is on a lead-in or lead-out. */
  audio: string;
}

export interface DumpMiss extends DumpEvent {
  forms_other_ayah: boolean;
  cause?: "evidence" | "tracker" | "checker";
  /** The checker's final state at this word for this class when it raised it but never confirmed it. */
  raisedAs?: string;
}

export interface DumpArmResult {
  /** Final states including suspected, uncertain and resolved reports, before context filtering. */
  engineEvents: DumpEvent[];
  hits: string[];
  misses: DumpMiss[];
  falsePositives: DumpEvent[];
  leadInWarnings?: DumpEvent[];
  leadOutWarnings?: DumpEvent[];
  gate: StartGate;
  /** Lead-in and lead-out arms: the stitched parts, in session time. */
  segments?: (Segment & { audio: string })[];
}

export interface DumpWord {
  at: string;
  text: string;
  /** Canonical phonemes. */
  ref: string;
  /** Decoded phonemes aligned to this word ("" when none). */
  said: string;
  /** Decoded time span in clip seconds, when anything aligned. */
  fromS?: number;
  toS?: number;
}

export interface DumpClip {
  id: string;
  speaker: string;
  device?: string;
  level?: string;
  kind: string;
  ayahs: string[];
  audio: string;
  durationS: number;
  /** Original source label, including unsupported classes and non-errors omitted from memorization scoring. */
  sourceLabel?: Record<string, unknown>;
  labels: (LabelEvent & { expected?: string; observed?: string })[];
  repeats: string[];
  words: DumpWord[];
  arms: Partial<Record<DumpArm, DumpArmResult>>;
}

export interface ClipDump {
  version: typeof CLIP_DUMP_VERSION;
  sha: string;
  listener: string;
  recordedAt: string;
  clips: DumpClip[];
  /** Local replay provenance; older dumps use explicitly marked benchmark defaults. */
  replay?: {
    helpEvidenceDir: string;
    helpEvidenceLayout?: "flat" | "wav";
    leadEvidenceDir?: string;
    referenceAsset: string;
    phraseAsset: string;
    settings: { chunkMs: number; acquireMs: number; minConfidence: number; confirmMs: number; localWindow: boolean; finaliseFree: boolean };
  };
}

const at = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const sig = (e: { errorClass: string; location: WordRef }) => `${e.errorClass}/${at(e.location)}`;
const ayahOf = (r: WordRef) => `${r.surah}:${r.ayah}`;
/** Checker events fire after the word; seek this far before the emission when the word has no decoded time. */
const EMIT_LEAD_S = 2.5;
const PRE_ROLL_S = 1;

/** The clip's written words with the greedy decode aligned to them. */
export function clipWords(index: ReferenceIndex, asset: ReferenceAsset, clip: AcceptanceClip): DumpWord[] {
  const text = new Map<string, string>();
  for (const a of asset.ayahs) a.w.forEach((w, i) => text.set(`${a.s}:${a.a}:${i + 1}`, w));
  const ws = index.words.filter((w) => clip.ayahs.includes(ayahOf(w)));
  if (!ws.length) return [];
  const lo = ws[0].start;
  const ref = index.phonemes.slice(lo, ws.at(-1)!.end);
  const ph = clip.evidence.phonemes;
  const aligned = alignRef(ref, ph.map((p) => p.p).join(""));
  const sec = (f: number) => (f * clip.evidence.frameMs) / 1000;
  return ws.map((w) => {
    const js = aligned.slice(w.start - lo, w.end - lo).filter((j): j is number => j !== null);
    const first = js.length ? Math.min(...js) : null, last = js.length ? Math.max(...js) : null;
    return {
      at: at(w),
      text: text.get(at(w)) ?? "",
      ref: ref.slice(w.start - lo, w.end - lo),
      said: first === null ? "" : ph.slice(first, last! + 1).map((p) => p.p).join(""),
      ...(first === null ? {} : { fromS: sec(ph[first].start), toS: sec(ph[last!].end) }),
    };
  });
}

interface ArmInput {
  score: ClipScore;
  /** For lead arms: the unrestricted score, the stitched segments and each segment's audio. */
  raw?: ClipScore;
  segments?: Segment[];
  segmentAudio?: string[];
  leadAyahs?: string[];
  followAyahs?: string[];
  /** Declared counterpart, for E0 attribution of free-arm misses. */
  declared?: ClipScore;
}

export function dumpArm(index: ReferenceIndex, clip: AcceptanceClip, words: DumpWord[], audio: string, o: ArmInput): DumpArmResult {
  const clipSeg = o.segments?.find((s) => s.source === clip.id);
  const offsetS = (clipSeg?.startMs ?? 0) / 1000;
  const wordAt = new Map(words.map((w) => [w.at, w]));
  const place = (e: { location: WordRef; timeMs?: number }): { seekS: number; audio: string } => {
    const w = wordAt.get(at(e.location));
    if (w?.fromS !== undefined) return { seekS: Math.max(0, w.fromS - PRE_ROLL_S), audio };
    const t = (e.timeMs ?? 0) / 1000;
    const seg = o.segments?.find((s) => s.ayahs.includes(ayahOf(e.location))) ?? o.segments?.find((s) => t * 1000 >= s.startMs && t * 1000 < s.endMs);
    if (seg && o.segments && o.segmentAudio) {
      return { seekS: Math.max(0, t - seg.startMs / 1000 - EMIT_LEAD_S), audio: o.segmentAudio[o.segments.indexOf(seg)] };
    }
    return { seekS: Math.max(0, t - offsetS - EMIT_LEAD_S), audio };
  };
  const ev = (e: CheckerEvent): DumpEvent => ({ errorClass: e.errorClass, at: at(e.location), state: e.state, expected: e.expected, observed: e.observed, confidence: e.confidence, checkerEvent: e, ...place(e) });
  const causes = o.declared ? new Map(attribute(index, clip, o.score, o.declared).map((a) => [sig(a.miss), a.cause])) : null;
  const finals = new Map(o.score.final.map((e) => [sig(e), e.state]));
  const missed = new Set(o.score.misses.map(sig));
  const onAyahs = (ayahs: string[] | undefined) => (e: CheckerEvent) => !!ayahs?.includes(ayahOf(e.location));
  return {
    engineEvents: (o.raw ?? o.score).final.map(ev),
    hits: clip.events.filter((e) => e.state === "confirmed" && !missed.has(sig(e)) && o.score.confirmed.some((c) => sig(c) === sig(e))).map(sig),
    misses: o.score.misses.map((m) => ({
      errorClass: m.errorClass, at: at(m.location), state: m.state, forms_other_ayah: !!m.forms_other_ayah,
      expected: (m as { expected?: string }).expected, observed: (m as { observed?: string }).observed,
      ...(causes ? { cause: causes.get(sig(m)) } : {}),
      ...(finals.has(sig(m)) ? { raisedAs: finals.get(sig(m)) } : {}),
      ...place(m),
    })),
    falsePositives: o.score.falsePositives.map(ev),
    ...(o.raw ? {
      leadInWarnings: o.raw.falsePositives.filter(onAyahs(o.leadAyahs)).map(ev),
      leadOutWarnings: o.raw.falsePositives.filter(onAyahs(o.followAyahs)).map(ev),
    } : {}),
    gate: o.score.gate,
    ...(o.segments && o.segmentAudio ? { segments: o.segments.map((s, i) => ({ ...s, audio: o.segmentAudio![i] })) } : {}),
  };
}
