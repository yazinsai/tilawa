import type { ReferenceIndex } from "../src/reference.js";

export interface TrainingLabel {
  id: string; speaker: string; split: string; kind: string; ayahs: string[]; wav: string;
  events?: { errorClass: string; location: { surah: number; ayah: number; word: number }; words?: number }[];
  non_errors?: { kind: string; location: { surah: number; ayah: number; word: number } }[];
  unsupported?: unknown[]; exclude?: unknown; excluded?: unknown; scored?: boolean;
}

/** Reject a mixed-split speaker as a unit. Scripted lexical substitutions lack a verified spoken-phone path. */
export function trainingLabels(rows: TrainingLabel[]): TrainingLabel[] {
  const held = new Set(rows.filter(r => r.split !== "train").map(r => r.speaker));
  return rows.filter(r => r.split === "train" && !held.has(r.speaker) && r.id && r.speaker && r.wav
    && r.ayahs?.length && !r.unsupported?.length && !r.exclude && !r.excluded && r.scored !== false
    && (r.events ?? []).every(e => ["omitted_word", "skipped_passage"].includes(e.errorClass))
    && (r.non_errors ?? []).every(e => e.kind === "repeat")
    && (r.kind === "none" ? !r.events?.length && !r.non_errors?.length : !!r.events?.length || !!r.non_errors?.length))
    .sort((a, b) => a.id.localeCompare(b.id));
}

/** Scripted omissions and repeats specify reference occurrences, including repeated visits to the same word. */
export function transcriptPath(index: ReferenceIndex, r: TrainingLabel): number[] {
  const path: number[] = [];
  for (let i = 0; i < index.words.length; i++) {
    const w = index.words[i];
    if (!r.ayahs.includes(`${w.surah}:${w.ayah}`) || w.end === w.start) continue;
    const omitted = (r.events ?? []).some(e => {
      const first = index.words.findIndex(v => v.surah === e.location.surah && v.ayah === e.location.ayah && v.word === e.location.word);
      return first >= 0 && i >= first && i < first + (e.words ?? 1);
    });
    if (omitted) continue;
    path.push(i);
    if ((r.non_errors ?? []).some(e => e.location.surah === w.surah && e.location.ayah === w.ayah && e.location.word === w.word)) path.push(i);
  }
  return path;
}

/** Transcript-only token boundaries for training splices and location labels; no tracker visits supply labels. */
export function wordBoundaries(index: ReferenceIndex, path: number[], heard: string): number[] {
  const ref = path.map(i => index.phonemes.slice(index.words[i].start, index.words[i].end)).join("");
  const m = ref.length, n = heard.length;
  const c = Array.from({ length: n + 1 }, () => new Float64Array(m + 1));
  const op = Array.from({ length: n + 1 }, () => new Uint8Array(m + 1));
  for (let j = 1; j <= m; j++) { c[0][j] = j; op[0][j] = 2; }
  for (let t = 1; t <= n; t++) {
    c[t][0] = t; op[t][0] = 1;
    for (let j = 1; j <= m; j++) {
      c[t][j] = c[t - 1][j - 1] + Number(ref[j - 1] !== heard[t - 1]);
      if (c[t - 1][j] + 1 < c[t][j]) { c[t][j] = c[t - 1][j] + 1; op[t][j] = 1; }
      if (c[t][j - 1] + 1 < c[t][j]) { c[t][j] = c[t][j - 1] + 1; op[t][j] = 2; }
    }
  }
  let t = n, j = m;
  const at = new Int32Array(m + 1).fill(-1); at[m] = n;
  while (t || j) {
    const k = op[t][j];
    if (k === 0) { t--; j--; } else if (k === 1) t--; else j--;
    at[j] = t;
  }
  const boundaries = [0];
  let offset = 0;
  for (const i of path) { offset += index.words[i].end - index.words[i].start; boundaries.push(at[offset]); }
  boundaries[boundaries.length - 1] = n;
  return boundaries;
}
