import { DEFAULT_CONFIG, type EngineConfig } from "./config.js";
import { QuranCorpus } from "./corpus.js";
import { alignSemiGlobal, normalizedDistance } from "./alignment.js";
import { UNKNOWN_ID, costTable, type CostTable } from "./phonemeCost.js";
import type { SearchHint, SearchHit, SearchResult, StripResult } from "./types.js";

export const ISTIADHA = "ءَعُۥۥذُبِللَااهِمِنَششَييطَاانِررَجِۦۦم";
export const BASMALA = "بِسمِللَااهِررَحمَاانِررَحِۦۦۦۦم";

const GRAM = 5;
const BUCKET_BITS = 18;
const BUCKETS = 1 << BUCKET_BITS;
const BUCKET_MASK = BUCKETS - 1;
const WINDOW_BITS = 5;
const MAX_POSTINGS = 400;
const CANDIDATE_WINDOWS = 24;
const VERIFY_MARGIN_BEFORE = 16;
const VERIFY_MARGIN_AFTER = 32;
const MIN_ALIGNED = 20;
const SURAH_GAP = 8;
const SHORT_QUERY = 100;
const PREAMBLE_MAX_DISTANCE = 0.3;
const GROWING_DISTANCE = 0.35;
const FNV_OFFSET = 2166136261;
const FNV_PRIME = 16777619;

export function fnv1aBucket(ids: ArrayLike<number>, at: number): number {
  let h = FNV_OFFSET | 0;
  for (let k = 0; k < GRAM; k++) {
    h = Math.imul(h ^ (ids[at + k]! | 0), FNV_PRIME);
  }
  return (h >>> 0) & BUCKET_MASK;
}

export function stripPreambles(query: string, table: CostTable): StripResult {
  let offset = 0;
  let basmala = false;
  let basmalaOffset = 0;
  for (const phrase of [ISTIADHA, BASMALA]) {
    const rest = query.length - offset;
    if (rest <= 0) break;
    const L = phrase.length;
    let bestLen = -1;
    let bestDist = Infinity;
    let bestTie = Infinity;
    const lo = Math.max(1, L - 4);
    const hi = Math.min(rest, L + 4);
    for (let len = lo; len <= hi; len++) {
      const slice = query.slice(offset, offset + len);
      const target = phrase.slice(0, Math.min(phrase.length, len));
      const d = normalizedDistance(table.encode(slice), table.encode(target), table);
      const tie = Math.abs(len - L);
      if (d < bestDist || (d === bestDist && tie < bestTie)) {
        bestDist = d;
        bestLen = len;
        bestTie = tie;
      }
    }
    if (bestLen >= 0 && bestDist <= PREAMBLE_MAX_DISTANCE) {
      if (phrase === BASMALA) {
        basmala = true;
        basmalaOffset = offset;
      }
      offset += bestLen;
    }
  }
  return { offset, basmala, basmalaOffset };
}

function growingIstiadha(query: string, table: CostTable): boolean {
  if (query.length > ISTIADHA.length + 4) return false;
  const target = ISTIADHA.slice(0, Math.min(ISTIADHA.length, query.length));
  return normalizedDistance(table.encode(query), table.encode(target), table) <= GROWING_DISTANCE;
}

/** True while `rest` could still grow into an isti'adha or basmala. */
export function preamblePending(rest: string, table: CostTable): boolean {
  if (!rest.length) return true;
  for (const phrase of [ISTIADHA, BASMALA]) {
    if (rest.length > phrase.length + 4) continue;
    const target = phrase.slice(0, Math.min(phrase.length, rest.length));
    if (normalizedDistance(table.encode(rest), table.encode(target), table) <= GROWING_DISTANCE) return true;
  }
  return false;
}

export class QuranIndex {
  readonly corpus: QuranCorpus;
  readonly table: CostTable;
  private readonly cfg: EngineConfig;
  private readonly sep: Uint8Array;
  private readonly surahSepStart: Int32Array;
  private readonly surahSepLen: Int32Array;
  private readonly surahCorpusStart: Int32Array;
  private readonly bucketStart: Int32Array;
  private readonly postings: Int32Array;

  constructor(corpus: QuranCorpus, cfg: EngineConfig = DEFAULT_CONFIG, table: CostTable = costTable()) {
    this.corpus = corpus;
    this.cfg = cfg;
    this.table = table;
    const nSurah = corpus.surahs.length;
    this.surahSepStart = new Int32Array(nSurah);
    this.surahSepLen = new Int32Array(nSurah);
    this.surahCorpusStart = new Int32Array(nSurah);
    let sepLen = 0;
    for (let i = 0; i < nSurah; i++) {
      const s = corpus.surahs[i]!;
      const cs = corpus.wordStart[s.firstWord]!;
      const ce = corpus.wordStart[s.endWord]!;
      this.surahCorpusStart[i] = cs;
      this.surahSepLen[i] = ce - cs;
      sepLen += ce - cs;
      if (i < nSurah - 1) sepLen += SURAH_GAP;
    }
    const sep = new Uint8Array(sepLen);
    let at = 0;
    for (let i = 0; i < nSurah; i++) {
      this.surahSepStart[i] = at;
      const s = corpus.surahs[i]!;
      const cs = corpus.wordStart[s.firstWord]!;
      const ce = corpus.wordStart[s.endWord]!;
      sep.set(table.encode(corpus.text.slice(cs, ce)), at);
      at += ce - cs;
      if (i < nSurah - 1) {
        sep.fill(UNKNOWN_ID, at, at + SURAH_GAP);
        at += SURAH_GAP;
      }
    }
    this.sep = sep;

    const counts = new Int32Array(BUCKETS);
    const last = sep.length - GRAM;
    for (let p = 0; p <= last; p++) counts[fnv1aBucket(sep, p)]++;
    const bucketStart = new Int32Array(BUCKETS + 1);
    for (let b = 0; b < BUCKETS; b++) bucketStart[b + 1] = bucketStart[b]! + counts[b]!;
    const postings = new Int32Array(bucketStart[BUCKETS]!);
    const cursor = bucketStart.slice();
    for (let p = 0; p <= last; p++) {
      const b = fnv1aBucket(sep, p);
      postings[cursor[b]!] = p;
      cursor[b]!++;
    }
    this.bucketStart = bucketStart;
    this.postings = postings;
  }

  search(query: string, hint?: SearchHint | null, limit = 3): SearchResult {
    if (query.length < this.cfg.searchMinChars) return { hits: [], decisive: false };
    if (growingIstiadha(query, this.table)) return { hits: [], decisive: false };

    const stripped = stripPreambles(query, this.table);
    const rest = query.slice(stripped.offset);
    if (rest.length < this.cfg.searchMinChars) return { hits: [], decisive: false };

    if (stripped.basmala) {
      const fromBasmala = query.slice(stripped.basmalaOffset);
      const inner = this.searchSlice(fromBasmala, hint, limit, 0);
      if (inner.decisive && inner.hits[0] && inner.hits[0].queryStart <= 2) {
        for (const h of inner.hits) h.queryStart += stripped.basmalaOffset;
        return inner;
      }
    }

    const result = this.searchSlice(rest, hint, limit, 0);
    let bonus = 0;
    const collapsed = new Set<number>();
    for (let i = 0; i < result.hits.length; i++) {
      const h = result.hits[i]!;
      if (
        stripped.basmala &&
        h.surah === 1 &&
        h.ayah === 2 &&
        h.word <= 3
      ) {
        h.surah = 1;
        h.ayah = 1;
        h.word = 0;
        h.wordIndex = 0;
        h.refOffset = 0;
        h.refEnd = 0;
        h.queryStart = 0;
        collapsed.add(i);
        if (i === 0) bonus = stripped.offset - stripped.basmalaOffset;
      }
    }
    result.decisive = this.isDecisive(result.hits, rest.length, bonus, hint);
    for (let i = 0; i < result.hits.length; i++) {
      const h = result.hits[i]!;
      if (collapsed.has(i)) h.queryStart = stripped.basmalaOffset;
      else h.queryStart += stripped.offset;
    }
    if (result.decisive || query.length <= SHORT_QUERY) return result;

    const tail = this.search(query.slice(-SHORT_QUERY), hint, limit);
    if (tail.decisive) {
      const add = query.length - SHORT_QUERY;
      for (const h of tail.hits) h.queryStart += add;
      return tail;
    }
    return result;
  }

  private searchSlice(
    query: string,
    hint: SearchHint | null | undefined,
    limit: number,
    alignedBonus: number,
  ): SearchResult {
    if (query.length < this.cfg.searchMinChars) return { hits: [], decisive: false };
    const qids = this.table.encode(query);
    const votes = new Map<number, number>();
    const last = qids.length - GRAM;
    for (let u = 0; u <= last; u++) {
      const b = fnv1aBucket(qids, u);
      const a = this.bucketStart[b]!;
      const z = this.bucketStart[b + 1]!;
      if (z - a > MAX_POSTINGS) continue;
      for (let i = a; i < z; i++) {
        const p = this.postings[i]!;
        const w = (p - u) >> WINDOW_BITS;
        votes.set(w, (votes.get(w) ?? 0) + 1);
      }
    }
    const windows = [...votes.entries()].sort((x, y) => y[1] - x[1] || x[0] - y[0]);
    const kept = windows.slice(0, CANDIDATE_WINDOWS);
    const verified: SearchHit[] = [];
    const sep = this.sep;
    for (const [w] of kept) {
      const start = w << WINDOW_BITS;
      const from = Math.max(0, start - VERIFY_MARGIN_BEFORE);
      const to = Math.min(sep.length, start + qids.length + VERIFY_MARGIN_AFTER);
      const al = alignSemiGlobal(qids, sep, from, to, this.table, 0.5);
      if (qids.length - al.queryStart < this.cfg.searchMinChars) continue;
      const refOffset = this.mapSep(al.refStart);
      const refEnd = this.mapSep(al.refEnd);
      if (refEnd <= refOffset) continue;
      const wordIndex = this.corpus.wordAt(refOffset);
      verified.push({
        surah: this.corpus.wordSurah[wordIndex]!,
        ayah: this.corpus.wordAyah[wordIndex]!,
        word: this.corpus.wordInAyah[wordIndex]!,
        wordIndex,
        refOffset,
        refEnd,
        queryStart: al.queryStart,
        distance: al.distance,
      });
    }
    verified.sort((a, b) => a.distance - b.distance || a.wordIndex - b.wordIndex);
    const hits: SearchHit[] = [];
    for (const h of verified) {
      if (hits.some((k) => !(h.refEnd <= k.refOffset || h.refOffset >= k.refEnd))) continue;
      hits.push(h);
      if (hits.length >= Math.max(limit, 8)) break;
    }
    this.applyHint(hits, hint);
    const out = hits.slice(0, limit);
    return {
      hits: out,
      decisive: this.isDecisive(out, query.length, alignedBonus, hint),
    };
  }

  private applyHint(hits: SearchHit[], hint?: SearchHint | null): void {
    if (!hint || hits.length === 0) return;
    const best = hits[0]!;
    const near = hits.filter((h) => h.distance <= best.distance + this.cfg.searchDecisiveMargin);
    const hinted = near.filter((h) => h.surah === hint.surah);
    if (hinted.length === 0) return;
    const target = this.corpus.hasAyah(hint.surah, hint.ayah)
      ? this.corpus.ayahFirstWord(hint.surah, hint.ayah)
      : 0;
    hinted.sort(
      (a, b) =>
        Math.abs(a.wordIndex - target) - Math.abs(b.wordIndex - target) ||
        a.wordIndex - b.wordIndex,
    );
    const chosen = hinted[0]!;
    const rest = hits.filter((h) => h !== chosen);
    hits.length = 0;
    hits.push(chosen, ...rest);
  }

  private isDecisive(
    hits: SearchHit[],
    queryLength: number,
    alignedBonus: number,
    hint?: SearchHint | null,
  ): boolean {
    const best = hits[0];
    if (!best) return false;
    const aligned = queryLength - best.queryStart + alignedBonus;
    if (best.distance > this.cfg.searchDecisiveDistance) return false;
    if (aligned < MIN_ALIGNED) return false;
    const rival = this.rival(hits, hint);
    if (!rival) return true;
    return rival.distance - best.distance >= this.cfg.searchDecisiveMargin;
  }

  private rival(hits: SearchHit[], hint?: SearchHint | null): SearchHit | undefined {
    const best = hits[0];
    if (!best) return undefined;
    if (hint) {
      const cap = best.distance + this.cfg.searchDecisiveMargin;
      const near = hits.filter((h) => h.distance <= cap);
      if (near.some((h) => h.surah === hint.surah)) {
        return hits.find((h) => h !== best && h.distance > cap);
      }
    }
    return hits[1];
  }

  private mapSep(sepOff: number): number {
    const starts = this.surahSepStart;
    let lo = 0;
    let hi = starts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (starts[mid]! <= sepOff) lo = mid;
      else hi = mid - 1;
    }
    const local = sepOff - starts[lo]!;
    const len = this.surahSepLen[lo]!;
    const cs = this.surahCorpusStart[lo]!;
    if (local >= len) return cs + len;
    if (local < 0) return cs;
    return cs + local;
  }
}
