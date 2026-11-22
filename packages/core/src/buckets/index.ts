import type { WordRef, WordSpan } from "../events.js";
import type { ReferenceIndex } from "../reference.js";
import type { ProposedLabel } from "../relabel/index.js";

/** A label by text alone: the clip is one of `occurrences`, and which one can't be told from the audio. */
export interface TextOnlyLabel {
  /** Canonical phonemes shared by every occurrence. */
  text: string;
  occurrences: WordSpan[];
}

const positions = new WeakMap<ReferenceIndex, Map<string, number>>();
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
function wordIndex(index: ReferenceIndex, r: WordRef): number {
  let m = positions.get(index);
  if (!m) {
    m = new Map(index.words.map((w, i) => [key(w), i]));
    positions.set(index, m);
  }
  const i = m.get(key(r));
  if (i === undefined) throw new Error(`No word ${key(r)} in ${index.version}`);
  return i;
}

export function spanPhonemes(index: ReferenceIndex, span: WordSpan): string {
  return index.phonemes.slice(index.words[wordIndex(index, span.start)].start, index.words[wordIndex(index, span.end)].end);
}

export const BUCKETS = ["clean-match", "deviant", "ambiguous", "reject"] as const;
export type Bucket = (typeof BUCKETS)[number];

export interface BucketCutoffs {
  /** Lowest tracker confidence that counts as located (ADR 0001). */
  minConfidence: number;
  /** PER against the located passage. */
  cleanMaxPer: number;
  /** Mean teacher phoneme confidence. */
  cleanMinTeacherConf: number;
  /** Above this PER the location is more likely wrong than the recitation deviant. */
  deviantMaxPer: number;
  rejectBelowTeacherConf: number;
}

/** Proposed from the 10% sample; not yet tuned on the hand-checked clips. */
export const PROPOSED_CUTOFFS: BucketCutoffs = {
  minConfidence: 0.2,
  cleanMaxPer: 0.03,
  cleanMinTeacherConf: 0.9,
  deviantMaxPer: 0.3,
  rejectBelowTeacherConf: 0.5,
};

/** Per-clip teacher statistics; null when the clip was never decoded. */
export interface TeacherStats {
  meanConf: number;
  durationS: number;
}

export type BucketLabel = { kind: "span"; span: WordSpan } | ({ kind: "text" } & TextOnlyLabel);

export type BucketReason = "match" | "mismatch" | "teacher_unsure" | "high_per" | "low_confidence" | "ambiguous" | "low_teacher_confidence" | "lost" | "insufficient_evidence" | "decode_failed" | "too_long";

export interface BucketedClip {
  bucket: Bucket;
  reason: BucketReason;
  /** Null in `ambiguous` and `reject`. */
  label: BucketLabel | null;
}

export function bucketClip(index: ReferenceIndex, l: ProposedLabel, teacher: TeacherStats | null, c: BucketCutoffs): BucketedClip {
  const reject = (reason: BucketReason): BucketedClip => ({ bucket: "reject", reason, label: null });
  if (l.exclusion && l.exclusion !== "ambiguous") return reject(l.exclusion);
  if (!teacher || teacher.meanConf < c.rejectBelowTeacherConf) return reject("low_teacher_confidence");

  let label: BucketLabel | null = null;
  if (l.proposed && (l.confidence ?? 0) >= c.minConfidence) label = { kind: "span", span: l.proposed };
  else if (l.exclusion === "ambiguous") {
    const t = textOnlyLabel(index, l);
    if (t) label = { kind: "text", ...t };
  }
  if (!label) return { bucket: "ambiguous", reason: l.proposed ? "low_confidence" : "ambiguous", label: null };

  const per = l.per ?? Infinity;
  if (per > c.deviantMaxPer) return reject("high_per");
  if (per > c.cleanMaxPer) return { bucket: "deviant", reason: "mismatch", label };
  if (teacher.meanConf < c.cleanMinTeacherConf) return { bucket: "deviant", reason: "teacher_unsure", label };
  return { bucket: "clean-match", reason: "match", label };
}

/** An `ambiguous` clip whose candidates all have the same canonical phonemes (ADR 0001). */
export function textOnlyLabel(index: ReferenceIndex, label: ProposedLabel): TextOnlyLabel | null {
  if (label.exclusion !== "ambiguous" || label.candidates.length < 2) return null;
  const texts = label.candidates.map((c) => spanPhonemes(index, c.span));
  if (texts.some((t) => t !== texts[0])) return null;
  return { text: texts[0], occurrences: label.candidates.map((c) => c.span) };
}

export const SPLITS = ["train", "val", "test"] as const;
export type Split = (typeof SPLITS)[number];

export interface ManifestClip extends BucketedClip {
  shard: string;
  row: number;
  path: string;
  durationS: number | null;
  /** Speaker cluster; null when the clip has no embedding, and then it's split on its own. */
  cluster: string | null;
}

export interface ManifestOptions {
  version: string;
  /** Filename stems (`surah_ayah_id`, no extension) to drop, e.g. Quran-Lab's `tlog_holdout`. */
  excludeStems: ReadonlySet<string>;
  fractions: { val: number; test: number };
  /** Clusters holding more than this share of clips always go to train. Default 0.01. */
  maxClusterShare?: number;
}

export interface Manifest {
  version: string;
  fractions: { val: number; test: number };
  maxClusterShare: number;
  counts: Record<Bucket, Record<Split, { clips: number; hours: number; clusters: number }>>;
  excluded: { benchmark: number };
}

/** FNV-1a, mapped to [0, 1). */
function unit(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193);
  return (h >>> 0) / 2 ** 32;
}

export const stemOf = (path: string) => path.replace(/^.*\//, "").replace(/\.\w+$/, "");

export function buildManifests(clips: ManifestClip[], o: ManifestOptions): { manifest: Manifest; files: Record<string, ManifestClip[]> } {
  const files: Record<string, ManifestClip[]> = {};
  const counts = Object.fromEntries(BUCKETS.map((b) => [b, Object.fromEntries(SPLITS.map((s) => [s, { clips: 0, hours: 0, clusters: 0 }]))])) as Manifest["counts"];
  const maxClusterShare = o.maxClusterShare ?? 0.01;
  const clusterKey = (c: ManifestClip) => c.cluster ?? `clip:${c.shard}/${c.row}`;
  const kept = clips.filter((c) => !o.excludeStems.has(stemOf(c.path)));
  const benchmark = clips.length - kept.length;

  // Whole clusters in a stable pseudo-random order fill test, then val; the rest, and any oversized cluster, is train.
  const sizes = new Map<string, number>();
  for (const c of kept) sizes.set(clusterKey(c), (sizes.get(clusterKey(c)) ?? 0) + 1);
  const splitOf = new Map<string, Split>();
  const filled = { test: 0, val: 0 };
  const total = kept.length;
  for (const [k, n] of [...sizes].sort((a, b) => unit(a[0]) - unit(b[0]) || (a[0] < b[0] ? -1 : 1))) {
    let split: Split = "train";
    if (n <= maxClusterShare * total) {
      if (filled.test < o.fractions.test * total) split = "test";
      else if (filled.val < o.fractions.val * total) split = "val";
    }
    if (split !== "train") filled[split] += n;
    splitOf.set(k, split);
  }

  const seen = new Set<string>();
  for (const c of kept) {
    const split = splitOf.get(clusterKey(c))!;
    (files[`${c.bucket}/${split}`] ??= []).push(c);
    const n = counts[c.bucket][split];
    n.clips++;
    n.hours += (c.durationS ?? 0) / 3600;
    const k = `${c.bucket}/${split}/${clusterKey(c)}`;
    if (!seen.has(k)) {
      seen.add(k);
      n.clusters++;
    }
  }
  return { manifest: { version: o.version, fractions: o.fractions, maxClusterShare, counts, excluded: { benchmark } }, files };
}
