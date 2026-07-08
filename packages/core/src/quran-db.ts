import { ratio, fragmentScore } from "./levenshtein.js";
import type { QuranVerse } from "./types.js";

const _BSM_PHONEMES_JOINED = "bismi allahi arraHmaani arraHiimi";
const _BSM_PHONEME_TOKENS = "b i s m i | a l l a h i | a r r a H m aa n i | a r r a H ii m i".split(
  " ",
);
const _BSM_ARABIC_WORDS = ["بسم", "الله", "الرحمن", "الرحيم"];

export interface QuranTokenEncoder {
  encodeRawPhonemes(rawPhonemes: string): number[];
}

export type QuranCtcTokenTable = Record<string, number[]>;

export interface QuranCandidate {
  surah: number;
  ayah: number;
  ayah_end?: number | null;
  text: string;
  phonemes_joined: string;
  phoneme_token_ids: number[];
  stage_a_score: number;
  raw_score: number;
  bonus: number;
  kind: "single" | "span";
  surah_rank?: number;
}

export interface QuranChampionMatch {
  surah: number;
  ayah: number;
  ayah_end?: number | null;
  text: string;
  phonemes_joined: string;
  score: number;
  raw_score: number;
  bonus: number;
  _prefix_rescue?: boolean;
  _global_span_rescue?: boolean;
  runners_up?: Array<{
    surah: number;
    ayah: number;
    raw_score: number;
    bonus: number;
    score: number;
    phonemes_joined: string;
  }>;
}

export interface CandidateRetrieval {
  singles: QuranCandidate[];
  spans: QuranCandidate[];
  combined: QuranCandidate[];
}

interface RetrievalOptions {
  maxSpan?: number;
  hint?: [number, number] | null;
  singleLimit?: number;
  topSurahs?: number;
  spanLimit?: number;
}

const JOINT_TOP_K_LEVENSHTEIN = 18;
const JOINT_TOP_SURAHS = 32;
const JOINT_MAX_SPAN = 6;
const JOINT_FRAGMENT_BLEND = 0.82;
const JOINT_PREFIX_MAX_SPAN = 7;
const JOINT_PREFIX_MIN_CHARS = 34;
const JOINT_PREFIX_MIN_SCORE = 0.50;
const JOINT_PREFIX_MARGIN = -0.02;
const JOINT_GLOBAL_SPAN_MIN_CHARS = 80;
const JOINT_GLOBAL_SPAN_MIN_SCORE = 0.54;
const JOINT_GLOBAL_SPAN_MARGIN = -0.015;
const JOINT_GLOBAL_SPAN_SHORTLIST = 320;
const JOINT_OPENING_COLLAPSE_MIN_CHARS = 34;
const JOINT_OPENING_COLLAPSE_MAX_CHARS = 115;
const JOINT_OPENING_COLLAPSE_MIN_SCORE = 0.50;
const JOINT_SHORT_OPENING_MAX_CHARS = 10;
const JOINT_SHORT_OPENING_MIN_QUERY_CHARS = 8;
const JOINT_SHORT_OPENING_MIN_SCORE = 0.48;
const JOINT_SHORT_OPENING_MARGIN = -0.18;

interface GlobalSpanRow {
  surah: number;
  ayah: number;
  ayah_end: number;
  phonemes: string;
  phonemesNs: string;
  bigrams: Set<string>;
  trigrams: Set<string>;
}

export function partialRatio(short: string, long: string): number {
  if (!short || !long) return 0.0;
  if (short.length > long.length) [short, long] = [long, short];
  const window = short.length;
  let best = 0.0;
  for (let i = 0; i <= Math.max(0, long.length - window); i++) {
    const r = ratio(short, long.slice(i, i + window));
    if (r > best) {
      best = r;
      if (best === 1.0) break;
    }
  }
  return best;
}

export class QuranDB {
  verses: QuranVerse[];
  private _byRef: Map<string, QuranVerse> = new Map();
  private _bySurah: Map<number, QuranVerse[]> = new Map();
  private _jointPrefixSpans: QuranChampionMatch[] | null = null;
  private _jointGlobalSpans: GlobalSpanRow[] | null = null;
  private tokenEncoder?: QuranTokenEncoder;
  private ctcTokenTable?: QuranCtcTokenTable;

  constructor(
    data: QuranVerse[],
    tokenEncoder?: QuranTokenEncoder,
    ctcTokenTable?: QuranCtcTokenTable,
  ) {
    this.tokenEncoder = tokenEncoder;
    this.ctcTokenTable = ctcTokenTable;
    this.verses = data;
    for (const v of data) {
      this._byRef.set(`${v.surah}:${v.ayah}`, v);
      const arr = this._bySurah.get(v.surah) ?? [];
      arr.push(v);
      this._bySurah.set(v.surah, arr);

      v.phoneme_tokens = v.phoneme_tokens?.length
        ? v.phoneme_tokens
        : v.phonemes.trim().split(/\s+/).filter(Boolean);

      if (
        v.ayah === 1 &&
        v.surah !== 1 &&
        v.surah !== 9 &&
        v.phonemes_joined.startsWith(_BSM_PHONEMES_JOINED)
      ) {
        const stripped = v.phonemes_joined.slice(_BSM_PHONEMES_JOINED.length).trim();
        v.phonemes_joined_no_bsm = stripped || null;

        let strippedTokens = v.phoneme_tokens.slice(_BSM_PHONEME_TOKENS.length);
        if (strippedTokens[0] === "|") {
          strippedTokens = strippedTokens.slice(1);
        }
        v.phoneme_tokens_no_bsm = strippedTokens.length ? strippedTokens : null;
      } else {
        v.phonemes_joined_no_bsm = null;
        v.phoneme_tokens_no_bsm = null;
      }

      if (
        v.ayah === 1 &&
        v.surah !== 1 &&
        v.surah !== 9 &&
        this._startsWithArabicBismillah(v.phoneme_words)
      ) {
        const strippedWords = v.phoneme_words.slice(_BSM_ARABIC_WORDS.length);
        v.phonemes_joined_no_bsm = strippedWords.length ? strippedWords.join(" ") : null;

        const bsmTokenEnd = v.word_token_ends?.[_BSM_ARABIC_WORDS.length - 1] ?? 0;
        v.phoneme_tokens_no_bsm =
          bsmTokenEnd > 0 && v.phoneme_tokens.length > bsmTokenEnd
            ? v.phoneme_tokens.slice(bsmTokenEnd)
            : null;
        v.phoneme_token_ids_no_bsm =
          bsmTokenEnd > 0 && (v.phoneme_token_ids?.length ?? 0) > bsmTokenEnd
            ? v.phoneme_token_ids!.slice(bsmTokenEnd)
            : null;
      }

      v.phonemes_joined_ns = v.phonemes_joined.replace(/ /g, "");
      v.phonemes_joined_no_bsm_ns = v.phonemes_joined_no_bsm
        ? v.phonemes_joined_no_bsm.replace(/ /g, "")
        : null;

      if (this.tokenEncoder) {
        v.phoneme_token_ids = this.tokenEncoder.encodeRawPhonemes(v.phonemes);
        v.phoneme_token_ids_no_bsm = v.phoneme_tokens_no_bsm
          ? this.tokenEncoder.encodeRawPhonemes(v.phoneme_tokens_no_bsm.join(" "))
          : null;
      } else {
        v.phoneme_token_ids = v.phoneme_token_ids ?? [];
        v.phoneme_token_ids_no_bsm = v.phoneme_token_ids_no_bsm ?? null;
      }

      v.word_token_ends = v.word_token_ends?.length
        ? v.word_token_ends
        : this._computeWordTokenEnds(v.phoneme_tokens);
    }
  }

  get totalVerses(): number {
    return this.verses.length;
  }

  get surahCount(): number {
    return this._bySurah.size;
  }

  getVerse(surah: number, ayah: number): QuranVerse | undefined {
    return this._byRef.get(`${surah}:${ayah}`);
  }

  getSurah(surah: number): QuranVerse[] {
    return this._bySurah.get(surah) ?? [];
  }

  getNextVerse(surah: number, ayah: number): QuranVerse | undefined {
    const verses = this._bySurah.get(surah) ?? [];
    for (let i = 0; i < verses.length; i++) {
      if (verses[i].ayah === ayah) {
        if (i + 1 < verses.length) return verses[i + 1];
        const nextSurah = this._bySurah.get(surah + 1) ?? [];
        return nextSurah[0];
      }
    }
    return undefined;
  }

  private _startsWithArabicBismillah(words: readonly string[] | undefined): boolean {
    if (!words || words.length <= _BSM_ARABIC_WORDS.length) return false;
    return _BSM_ARABIC_WORDS.every((word, index) => words[index] === word);
  }

  /** Return candidates for verses whose non-Bsm phoneme token IDs are short (≤ maxTokens). */
  getShortVerseCandidates(maxTokens = 15): QuranCandidate[] {
    const result: QuranCandidate[] = [];
    for (const v of this.verses) {
      const ids = v.phoneme_token_ids_no_bsm ?? v.phoneme_token_ids ?? [];
      if (ids.length === 0 || ids.length > maxTokens) continue;
      result.push({
        surah: v.surah,
        ayah: v.ayah,
        text: v.phonemes_joined,
        phonemes_joined: v.phonemes_joined,
        phoneme_token_ids: ids,
        stage_a_score: 0,
        raw_score: 0,
        bonus: 0,
        kind: "single",
      });
    }
    return result;
  }

  search(text: string, topK = 5): (QuranVerse & { score: number })[] {
    const scored: (QuranVerse & { score: number })[] = [];
    for (const v of this.verses) {
      const score = ratio(text, v.phonemes_joined);
      scored.push({ ...v, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  retrieveCandidates(
    text: string,
    {
      maxSpan = 4,
      hint = null,
      singleLimit = 32,
      topSurahs = 3,
      spanLimit = 32,
    }: RetrievalOptions = {},
  ): CandidateRetrieval {
    if (!text.trim()) {
      return { singles: [], spans: [], combined: [] };
    }

    const bonuses = this._continuationBonuses(hint);
    const textWords = text.split(/\s+/).filter(Boolean);
    const noSpaceText = text.replace(/ /g, "");
    const scored: [QuranVerse, number, number, number][] = [];

    for (const v of this.verses) {
      let raw = ratio(text, v.phonemes_joined);
      const verseWords = v.phoneme_words;
      const sharedWordCount = Math.min(textWords.length, verseWords.length);
      if (sharedWordCount > 0) {
        const textPrefix = textWords.slice(0, sharedWordCount).join(" ");
        const versePrefix = verseWords.slice(0, sharedWordCount).join(" ");
        raw = Math.max(raw, ratio(textPrefix, versePrefix));
      }
      if (noSpaceText.length <= 10) {
        raw = Math.max(raw, this._shortQueryBoost(noSpaceText, v));
      }
      if (v.phonemes_joined_no_bsm) {
        raw = Math.max(raw, ratio(text, v.phonemes_joined_no_bsm));
        if (noSpaceText.length <= 10) {
          raw = Math.max(raw, this._shortQueryBoost(noSpaceText, v, true));
        }
      }
      const bonus = bonuses.get(`${v.surah}:${v.ayah}`) ?? 0.0;
      if (bonus > 0) {
        const sp = QuranDB._suffixPrefixScore(text, v.phonemes_joined);
        raw = Math.max(raw, sp);
      }
      scored.push([v, raw, bonus, Math.min(raw + bonus, 1.0)]);
    }
    scored.sort((a, b) => b[3] - a[3]);

    const pass2Surahs: number[] = [];
    for (let i = 0; i < scored.length && pass2Surahs.length < topSurahs; i++) {
      const surah = scored[i][0].surah;
      if (!pass2Surahs.includes(surah)) {
        pass2Surahs.push(surah);
      }
    }

    if (noSpaceText.length >= 8) {
      let resorted = false;
      for (let i = 0; i < scored.length; i++) {
        const [v, raw, bonus] = scored[i];
        if (noSpaceText.length >= (v.phonemes_joined_ns?.length ?? 0) * 0.8) continue;

        let frag = fragmentScore(noSpaceText, v.phonemes_joined_ns ?? "");
        if (v.phonemes_joined_no_bsm_ns) {
          frag = Math.max(frag, fragmentScore(noSpaceText, v.phonemes_joined_no_bsm_ns));
        }
        if (frag > raw) {
          const boosted = raw + (frag - raw) * 0.7;
          scored[i] = [v, boosted, bonus, Math.min(boosted + bonus, 1.0)];
          resorted = true;
        }
      }
      if (resorted) scored.sort((a, b) => b[3] - a[3]);
    }

    const singles = scored
      .slice(0, singleLimit)
      .map(([v, raw, bonus, total]) =>
        this._candidateFromVerse(v, raw, bonus, total),
      );

    const spans: QuranCandidate[] = [];
    for (let surahRank = 0; surahRank < pass2Surahs.length; surahRank++) {
      const surah = pass2Surahs[surahRank];
      const verses = this._bySurah.get(surah) ?? [];
      for (let i = 0; i < verses.length; i++) {
        for (let span = 2; span <= maxSpan; span++) {
          if (i + span > verses.length) break;
          const chunk = verses.slice(i, i + span);
          const spanText = this._joinedSpanPhonemes(chunk);
          let raw = ratio(text, spanText);
          const spanWords = spanText.split(/\s+/).filter(Boolean);
          const sharedWordCount = Math.min(textWords.length, spanWords.length);
          if (sharedWordCount > 0) {
            const textPrefix = textWords.slice(0, sharedWordCount).join(" ");
            const spanPrefix = spanWords.slice(0, sharedWordCount).join(" ");
            raw = Math.max(raw, ratio(textPrefix, spanPrefix));
          }
          const bonus = bonuses.get(`${chunk[0].surah}:${chunk[0].ayah}`) ?? 0.0;
          const score = Math.min(raw + bonus, 1.0);
          spans.push(this._candidateFromSpan(chunk, raw, bonus, score, surahRank));
        }
      }
    }
    spans.sort((a, b) => b.stage_a_score - a.stage_a_score);

    return {
      singles,
      spans: spans.slice(0, spanLimit),
      combined: singles.concat(spans.slice(0, spanLimit)),
    };
  }

  matchVerse(
    text: string,
    threshold = 0.3,
    maxSpan = 3,
    hint: [number, number] | null = null,
    returnTopK = 0,
  ): QuranChampionMatch | null {
    const retrieved = this.retrieveCandidates(text, {
      maxSpan,
      hint,
      singleLimit: Math.max(returnTopK, 5),
      topSurahs: 20,
      spanLimit: 64,
    });

    const ranked = retrieved.combined
      .slice()
      .sort((a, b) => b.stage_a_score - a.stage_a_score);
    const best = ranked[0];
    if (!best || best.stage_a_score < threshold) {
      return null;
    }

    const result: QuranChampionMatch = {
      surah: best.surah,
      ayah: best.ayah,
      ayah_end: best.ayah_end,
      text: best.text,
      phonemes_joined: best.phonemes_joined,
      score: best.stage_a_score,
      raw_score: best.raw_score,
      bonus: best.bonus,
    };
    if (returnTopK > 0) {
      result.runners_up = retrieved.singles.slice(0, returnTopK).map((candidate) => ({
        surah: candidate.surah,
        ayah: candidate.ayah,
        raw_score: Math.round(candidate.raw_score * 1000) / 1000,
        bonus: Math.round(candidate.bonus * 1000) / 1000,
        score: Math.round(candidate.stage_a_score * 1000) / 1000,
        phonemes_joined: candidate.phonemes_joined.slice(0, 60),
      }));
    }
    return result;
  }

  matchPhonemeTextJoint03(text: string, topK = JOINT_TOP_K_LEVENSHTEIN): QuranChampionMatch[] {
    return this._joint02MatchPhonemeText(text, topK);
  }

  bestJoint03Match(text: string): QuranChampionMatch | null {
    const top = this._joint02MatchPhonemeText(text, JOINT_TOP_K_LEVENSHTEIN);
    if (!top.length) return null;

    const best = top[0];
    const bestScore = best.score;
    const shortOpening = this._jointShortOpeningSpanCandidate(text, bestScore);
    if (shortOpening) {
      return shortOpening;
    }

    const bestIsLateSpan = best.ayah_end != null && best.ayah > 1;
    const lowConfidence = bestScore < 0.62;
    if (!bestIsLateSpan && !lowConfidence) return best;

    const noSpaceLen = text.replace(/ /g, "").length;
    const prefix = this._jointSurahPrefixCandidates(text);
    const globalSpan = this._jointGlobalSpanCandidates(text);
    const candidates = [best]
      .concat(prefix.filter((p) => p.score >= bestScore + JOINT_PREFIX_MARGIN))
      .concat(globalSpan.filter((g) => g.score >= bestScore + JOINT_GLOBAL_SPAN_MARGIN));
    candidates.sort((a, b) => b.score - a.score);
    const chosen = candidates[0];

    if (
      noSpaceLen >= JOINT_OPENING_COLLAPSE_MIN_CHARS &&
      noSpaceLen <= JOINT_OPENING_COLLAPSE_MAX_CHARS &&
      best.ayah_end != null &&
      best.ayah > 1
    ) {
      const sameSurahPrefix = prefix
        .filter((p) =>
          p.surah === best.surah &&
          p.score >= JOINT_OPENING_COLLAPSE_MIN_SCORE &&
          (p.ayah_end == null || best.ayah_end == null || p.ayah_end >= best.ayah_end),
        )
        .sort((a, b) => b.score - a.score);
      if (sameSurahPrefix.length > 0) return sameSurahPrefix[0];
    }

    return chosen;
  }

  bestJoint03MatchForHypotheses(
    hypotheses: readonly string[],
  ): { match: QuranChampionMatch; transcript: string } | null {
    let best: { match: QuranChampionMatch; transcript: string } | null = null;
    for (const transcript of hypotheses) {
      const match = this.bestJoint03Match(transcript);
      if (!match) continue;
      if (!best || match.score > best.match.score) {
        best = { match, transcript };
      }
    }
    return best;
  }

  private _joint02MatchPhonemeText(
    phonemeText: string,
    topK = JOINT_TOP_K_LEVENSHTEIN,
  ): QuranChampionMatch[] {
    if (!phonemeText.trim()) return [];

    const noSpaceText = phonemeText.replace(/ /g, "");
    const scored: [QuranVerse, number, number][] = [];
    for (const verse of this._jointCandidateVerses(noSpaceText)) {
      const ref = verse.phonemes_joined;
      if (!ref) continue;

      let raw = ratio(phonemeText, ref);
      if (noSpaceText.length <= 10) {
        raw = Math.max(raw, this._shortQueryBoost(noSpaceText, verse));
      }
      const noBsm = verse.phonemes_joined_no_bsm;
      if (noBsm) {
        raw = Math.max(raw, ratio(phonemeText, noBsm));
        if (noSpaceText.length <= 10) {
          raw = Math.max(raw, this._shortQueryBoost(noSpaceText, verse, true));
        }
      }
      scored.push([verse, raw, raw]);
    }
    scored.sort((a, b) => b[2] - a[2]);

    const pass2Surahs: number[] = [];
    for (const [verse] of scored) {
      if (!pass2Surahs.includes(verse.surah)) {
        pass2Surahs.push(verse.surah);
      }
      if (pass2Surahs.length >= JOINT_TOP_SURAHS) break;
    }

    if (noSpaceText.length >= 8) {
      let resorted = false;
      for (let i = 0; i < scored.length; i++) {
        const [verse, raw] = scored[i];
        const refNs = verse.phonemes_joined_ns ?? "";
        if (!refNs || noSpaceText.length >= refNs.length * 0.8) continue;

        let frag = fragmentScore(noSpaceText, refNs);
        const noBsmNs = verse.phonemes_joined_no_bsm_ns;
        if (noBsmNs) {
          frag = Math.max(frag, fragmentScore(noSpaceText, noBsmNs));
        }
        if (frag > raw) {
          const boosted = raw + (frag - raw) * JOINT_FRAGMENT_BLEND;
          scored[i] = [verse, boosted, boosted];
          resorted = true;
        }
      }
      if (resorted) scored.sort((a, b) => b[2] - a[2]);
    }

    const spanResults: QuranChampionMatch[] = [];
    for (const surahNum of pass2Surahs) {
      const verses = this._bySurah.get(surahNum) ?? [];
      for (let i = 0; i < verses.length; i++) {
        for (let span = 2; span <= JOINT_MAX_SPAN; span++) {
          if (i + span > verses.length) break;
          const chunk = verses.slice(i, i + span);
          const spanPhonemes = this._joinedSpanPhonemes(chunk);
          const score = QuranDB._round4(ratio(phonemeText, spanPhonemes));
          spanResults.push({
            surah: surahNum,
            ayah: chunk[0].ayah,
            ayah_end: chunk[chunk.length - 1].ayah,
            text: chunk.map((verse) => verse.text_uthmani).join(" "),
            phonemes_joined: spanPhonemes,
            score,
            raw_score: score,
            bonus: 0,
          });
        }
      }
    }

    const singles: QuranChampionMatch[] = scored.slice(0, Math.max(topK, 32)).map(([verse, raw, boosted]) => {
      const score = QuranDB._round4(boosted);
      return {
        surah: verse.surah,
        ayah: verse.ayah,
        ayah_end: null,
        text: verse.text_uthmani,
        phonemes_joined: verse.phonemes_joined,
        score,
        raw_score: QuranDB._round4(raw),
        bonus: 0,
      };
    });

    return singles
      .concat(spanResults)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  private _jointSurahPrefixCandidates(phonemeText: string): QuranChampionMatch[] {
    if (!phonemeText.trim()) return [];
    const noSpaceText = phonemeText.replace(/ /g, "");
    if (noSpaceText.length < JOINT_PREFIX_MIN_CHARS) return [];

    const out: QuranChampionMatch[] = [];
    for (const row of this._jointPrefixSpanTable()) {
      const raw = ratio(phonemeText, row.phonemes_joined);
      const frag = fragmentScore(noSpaceText, row.phonemes_joined.replace(/ /g, ""));
      const score = Math.max(raw, raw + (frag - raw) * JOINT_FRAGMENT_BLEND);
      if (score < JOINT_PREFIX_MIN_SCORE) continue;
      out.push({
        ...row,
        score: QuranDB._round4(score),
        raw_score: QuranDB._round4(raw),
        bonus: 0,
        _prefix_rescue: true,
      });
    }
    return out.sort((a, b) => b.score - a.score).slice(0, 12);
  }

  private _jointGlobalSpanCandidates(phonemeText: string): QuranChampionMatch[] {
    if (!phonemeText.trim()) return [];
    const noSpaceText = phonemeText.replace(/ /g, "");
    if (noSpaceText.length < JOINT_GLOBAL_SPAN_MIN_CHARS) return [];

    const qb = QuranDB._jointNgrams(noSpaceText, 2);
    const qt = QuranDB._jointNgrams(noSpaceText, 3);
    const rough: [number, GlobalSpanRow][] = [];
    for (const row of this._jointGlobalSpanTable()) {
      const ov =
        QuranDB._intersectionSize(qb, row.bigrams) +
        0.48 * QuranDB._intersectionSize(qt, row.trigrams);
      if (ov > 0) rough.push([ov, row]);
    }
    rough.sort((a, b) => b[0] - a[0]);

    const out: QuranChampionMatch[] = [];
    for (const [, row] of rough.slice(0, JOINT_GLOBAL_SPAN_SHORTLIST)) {
      const raw = ratio(phonemeText, row.phonemes);
      const frag = fragmentScore(noSpaceText, row.phonemesNs);
      const score = Math.max(raw, raw + (frag - raw) * JOINT_FRAGMENT_BLEND);
      if (score < JOINT_GLOBAL_SPAN_MIN_SCORE) continue;
      out.push({
        surah: row.surah,
        ayah: row.ayah,
        ayah_end: row.ayah_end,
        text: this._spanText(row.surah, row.ayah, row.ayah_end),
        phonemes_joined: row.phonemes,
        score: QuranDB._round4(score),
        raw_score: QuranDB._round4(raw),
        bonus: 0,
        _global_span_rescue: true,
      });
    }
    return out.sort((a, b) => b.score - a.score).slice(0, 12);
  }

  private _jointShortOpeningSpanCandidate(
    phonemeText: string,
    bestScore: number,
  ): QuranChampionMatch | null {
    const noSpaceText = phonemeText.replace(/ /g, "");
    if (noSpaceText.length < JOINT_SHORT_OPENING_MIN_QUERY_CHARS) return null;

    let best: QuranChampionMatch | null = null;
    for (const row of this._jointPrefixSpanTable()) {
      const first = this.getVerse(row.surah, 1);
      const second = this.getVerse(row.surah, 2);
      if (!first || !second) continue;
      const firstNs =
        first.phonemes_joined_no_bsm_ns ??
        first.phonemes_joined_ns ??
        first.phonemes_joined.replace(/ /g, "");
      if (!firstNs || firstNs.length > JOINT_SHORT_OPENING_MAX_CHARS) continue;
      if (!noSpaceText.startsWith(firstNs)) continue;
      const remainder = noSpaceText.slice(firstNs.length);
      const secondNs =
        second.phonemes_joined_ns ?? second.phonemes_joined.replace(/ /g, "");
      const compareLen = Math.min(secondNs.length, remainder.length, 12);
      if (
        compareLen >= 4 &&
        ratio(remainder.slice(0, compareLen), secondNs.slice(0, compareLen)) < 0.72
      ) {
        continue;
      }

      const raw = ratio(phonemeText, row.phonemes_joined);
      const frag = fragmentScore(noSpaceText, row.phonemes_joined.replace(/ /g, ""));
      const score = Math.max(raw, raw + (frag - raw) * JOINT_FRAGMENT_BLEND);
      if (
        score >= JOINT_SHORT_OPENING_MIN_SCORE &&
        score >= bestScore + JOINT_SHORT_OPENING_MARGIN &&
        (!best || score > best.score)
      ) {
        best = {
          ...row,
          score: QuranDB._round4(score),
          raw_score: QuranDB._round4(raw),
          bonus: 0,
          _prefix_rescue: true,
        };
      }
    }

    return best;
  }

  private _jointCandidateVerses(noSpaceText: string, maxCandidates = 950): QuranVerse[] {
    if (noSpaceText.length < 4) return this.verses;

    const qb = QuranDB._jointNgrams(noSpaceText, 2);
    const qt = QuranDB._jointNgrams(noSpaceText, 3);
    if (qb.size === 0 && qt.size === 0) return this.verses;

    const scored: [number, number][] = [];
    for (let i = 0; i < this.verses.length; i++) {
      const refNs = this.verses[i].phonemes_joined_ns ?? "";
      if (refNs.length < 2) continue;
      const ov =
        QuranDB._intersectionSize(qb, QuranDB._jointNgrams(refNs, 2)) +
        0.48 * QuranDB._intersectionSize(qt, QuranDB._jointNgrams(refNs, 3));
      if (ov > 0) scored.push([ov, i]);
    }
    if (scored.length < 80) return this.verses;
    scored.sort((a, b) => b[0] - a[0]);
    return scored.slice(0, maxCandidates).map(([, index]) => this.verses[index]);
  }

  private _jointPrefixSpanTable(): QuranChampionMatch[] {
    if (this._jointPrefixSpans) return this._jointPrefixSpans;

    const spans: QuranChampionMatch[] = [];
    for (const [surahNum, verses] of this._bySurah.entries()) {
      if (!verses.length || verses[0].ayah !== 1) continue;
      const maxSpan = Math.min(JOINT_PREFIX_MAX_SPAN, verses.length);
      for (let span = 2; span <= maxSpan; span++) {
        const chunk = verses.slice(0, span);
        spans.push({
          surah: surahNum,
          ayah: 1,
          ayah_end: chunk[chunk.length - 1].ayah,
          text: chunk.map((verse) => verse.text_uthmani).join(" "),
          phonemes_joined: this._joinedSpanPhonemes(chunk),
          score: 0,
          raw_score: 0,
          bonus: 0,
        });
      }
    }
    this._jointPrefixSpans = spans;
    return spans;
  }

  private _jointGlobalSpanTable(): GlobalSpanRow[] {
    if (this._jointGlobalSpans) return this._jointGlobalSpans;

    const spans: GlobalSpanRow[] = [];
    for (const [surahNum, verses] of this._bySurah.entries()) {
      for (let i = 0; i < verses.length; i++) {
        const maxSpan = Math.min(JOINT_PREFIX_MAX_SPAN, verses.length - i);
        for (let span = 2; span <= maxSpan; span++) {
          const chunk = verses.slice(i, i + span);
          const phonemes = this._joinedSpanPhonemes(chunk);
          const phonemesNs = phonemes.replace(/ /g, "");
          spans.push({
            surah: surahNum,
            ayah: chunk[0].ayah,
            ayah_end: chunk[chunk.length - 1].ayah,
            phonemes,
            phonemesNs,
            bigrams: QuranDB._jointNgrams(phonemesNs, 2),
            trigrams: QuranDB._jointNgrams(phonemesNs, 3),
          });
        }
      }
    }
    this._jointGlobalSpans = spans;
    return spans;
  }

  private _spanText(surah: number, ayah: number, ayahEnd: number): string {
    const verses = this._bySurah.get(surah) ?? [];
    return verses
      .filter((verse) => verse.ayah >= ayah && verse.ayah <= ayahEnd)
      .map((verse) => verse.text_uthmani)
      .join(" ");
  }

  private _computeWordTokenEnds(tokens: readonly string[]): number[] {
    const ends: number[] = [];
    let rawTokenIndex = 0;
    let lastWasBoundary = true;

    for (const token of tokens) {
      rawTokenIndex++;
      if (token === "|") {
        lastWasBoundary = true;
        continue;
      }
      if (lastWasBoundary) {
        ends.push(rawTokenIndex);
      } else {
        ends[ends.length - 1] = rawTokenIndex;
      }
      lastWasBoundary = false;
    }

    return ends;
  }

  private _candidateFromVerse(
    verse: QuranVerse,
    raw: number,
    bonus: number,
    total: number,
  ): QuranCandidate {
    return {
      surah: verse.surah,
      ayah: verse.ayah,
      ayah_end: verse.ayah,
      text: verse.text_uthmani,
      phonemes_joined: verse.phonemes_joined,
      phoneme_token_ids:
        verse.phoneme_token_ids_no_bsm ?? verse.phoneme_token_ids ?? [],
      stage_a_score: total,
      raw_score: raw,
      bonus,
      kind: "single",
    };
  }

  private _candidateFromSpan(
    chunk: QuranVerse[],
    raw: number,
    bonus: number,
    total: number,
    surahRank: number,
  ): QuranCandidate {
    const first = chunk[0];
    const spanKey = `${first.surah}:${first.ayah}:${chunk[chunk.length - 1].ayah}`;
    const tokenIds = this.ctcTokenTable?.[spanKey] ?? this._concatenatedSpanTokenIds(chunk);

    return {
      surah: first.surah,
      ayah: first.ayah,
      ayah_end: chunk[chunk.length - 1].ayah,
      text: chunk.map((verse) => verse.text_uthmani).join(" "),
      phonemes_joined: this._joinedSpanPhonemes(chunk),
      phoneme_token_ids: tokenIds,
      stage_a_score: total,
      raw_score: raw,
      bonus,
      kind: "span",
      surah_rank: surahRank,
    };
  }

  private _joinedSpanPhonemes(chunk: QuranVerse[]): string {
    const firstText = chunk[0].phonemes_joined_no_bsm ?? chunk[0].phonemes_joined;
    return [firstText]
      .concat(chunk.slice(1).map((verse) => verse.phonemes_joined))
      .join(" ");
  }

  private _concatenatedSpanTokenIds(chunk: QuranVerse[]): number[] {
    const tokenIds: number[] = [];
    const firstIds = chunk[0].phoneme_token_ids_no_bsm ?? chunk[0].phoneme_token_ids ?? [];
    tokenIds.push(...firstIds);
    for (let i = 1; i < chunk.length; i++) {
      tokenIds.push(...(chunk[i].phoneme_token_ids ?? []));
    }
    return tokenIds;
  }

  private _shortQueryBoost(
    noSpaceText: string,
    verse: QuranVerse,
    useNoBsm = false,
  ): number {
    const candidate = useNoBsm
      ? verse.phonemes_joined_no_bsm_ns ?? verse.phonemes_joined_ns ?? ""
      : verse.phonemes_joined_ns ?? "";
    if (!candidate) return 0;

    const prefixWindow = Math.min(candidate.length, noSpaceText.length + 6);
    const prefix = ratio(noSpaceText, candidate.slice(0, prefixWindow));
    const firstWord = useNoBsm
      ? (verse.phonemes_joined_no_bsm ?? "").split(" ")[0] ?? ""
      : verse.phoneme_words[0] ?? "";
    const firstWordScore = firstWord ? ratio(noSpaceText, firstWord) : 0;
    return Math.max(prefix, firstWordScore);
  }

  private _continuationBonuses(
    hint: [number, number] | null,
  ): Map<string, number> {
    const bonuses = new Map<string, number>();
    if (!hint) return bonuses;

    const [hSurah, hAyah] = hint;
    const nv = this._byRef.get(`${hSurah}:${hAyah + 1}`);
    if (nv) {
      bonuses.set(`${hSurah}:${hAyah + 1}`, 0.22);
      if (this._byRef.has(`${hSurah}:${hAyah + 2}`))
        bonuses.set(`${hSurah}:${hAyah + 2}`, 0.12);
      if (this._byRef.has(`${hSurah}:${hAyah + 3}`))
        bonuses.set(`${hSurah}:${hAyah + 3}`, 0.06);
    } else {
      const nextVerses = this._bySurah.get(hSurah + 1) ?? [];
      const bonusValues = [0.22, 0.12, 0.06];
      for (let i = 0; i < Math.min(nextVerses.length, 3); i++) {
        bonuses.set(
          `${nextVerses[i].surah}:${nextVerses[i].ayah}`,
          bonusValues[i],
        );
      }
    }
    return bonuses;
  }

  private static _suffixPrefixScore(text: string, verseText: string): number {
    const wordsT = text.split(" ");
    const wordsV = verseText.split(" ");
    if (wordsT.length < 2 || wordsV.length < 2) return 0.0;

    let best = 0.0;
    const maxTrim = Math.min(Math.floor(wordsT.length / 2), 4);
    for (let trim = 1; trim <= maxTrim; trim++) {
      const suffix = wordsT.slice(trim).join(" ");
      const n = wordsT.length - trim;
      const prefix = wordsV.slice(0, Math.min(n, wordsV.length)).join(" ");
      best = Math.max(best, ratio(suffix, prefix));
    }
    return best;
  }

  private static _jointNgrams(s: string, n: number): Set<string> {
    const out = new Set<string>();
    if (s.length < n) return out;
    for (let i = 0; i <= s.length - n; i++) {
      out.add(s.slice(i, i + n));
    }
    return out;
  }

  private static _intersectionSize(a: Set<string>, b: Set<string>): number {
    let count = 0;
    for (const item of a) {
      if (b.has(item)) count++;
    }
    return count;
  }

  private static _round4(value: number): number {
    return Math.round(value * 10000) / 10000;
  }
}
