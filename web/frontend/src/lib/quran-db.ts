import { ratio, fragmentScore } from "./levenshtein";
import type { QuranVerse } from "./types";

const _BSM_PHONEMES_JOINED = "bismi allahi arraHmaani arraHiimi";
const _BSM_PHONEME_TOKENS = "b i s m i | a l l a h i | a r r a H m aa n i | a r r a H ii m i".split(
  " ",
);

export interface QuranTokenEncoder {
  encodeRawPhonemes(rawPhonemes: string): number[];
}

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

  constructor(
    data: QuranVerse[],
    private tokenEncoder?: QuranTokenEncoder,
  ) {
    this.verses = data;
    for (const v of data) {
      this._byRef.set(`${v.surah}:${v.ayah}`, v);
      const arr = this._bySurah.get(v.surah) ?? [];
      arr.push(v);
      this._bySurah.set(v.surah, arr);

      v.phoneme_tokens = v.phonemes.trim().split(/\s+/).filter(Boolean);

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
        v.phoneme_token_ids = [];
        v.phoneme_token_ids_no_bsm = null;
      }

      v.word_token_ends = this._computeWordTokenEnds(v.phoneme_tokens);
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
  ): Record<string, any> | null {
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

    const result: Record<string, any> = {
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
    const tokenIds: number[] = [];
    const firstIds = first.phoneme_token_ids_no_bsm ?? first.phoneme_token_ids ?? [];
    tokenIds.push(...firstIds);
    for (let i = 1; i < chunk.length; i++) {
      tokenIds.push(...(chunk[i].phoneme_token_ids ?? []));
    }

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
}
