import { normalizeArabic } from "./normalizer.js";

export interface TextCTCResult {
  text: string;
  rawPhonemes: string;
  tokenIds: number[];
}

const WORD_PREFIX = "\u2581";

export class TextCTCDecoder {
  private vocab: Map<number, string>;
  private blankId: number;

  constructor(vocabJson: Record<string, string>, blankId?: number) {
    this.vocab = new Map();
    let maxId = 0;
    for (const [id, token] of Object.entries(vocabJson)) {
      const numId = Number(id);
      this.vocab.set(numId, token);
      maxId = Math.max(maxId, numId);
    }
    this.blankId = blankId ?? maxId;
  }

  decode(logprobs: Float32Array, timeSteps: number, vocabSize: number): TextCTCResult {
    const frameIds: number[] = [];
    for (let t = 0; t < timeSteps; t++) {
      const offset = t * vocabSize;
      let maxIdx = 0;
      let maxVal = logprobs[offset];
      for (let v = 1; v < vocabSize; v++) {
        const value = logprobs[offset + v];
        if (value > maxVal) {
          maxVal = value;
          maxIdx = v;
        }
      }
      frameIds.push(maxIdx);
    }

    const tokenIds: number[] = [];
    let previous = -1;
    for (const id of frameIds) {
      if (id !== previous && id !== this.blankId) {
        tokenIds.push(id);
      }
      previous = id;
    }

    const text = this.tokenIdsToText(tokenIds);
    return {
      text,
      rawPhonemes: text,
      tokenIds,
    };
  }

  getBlankId(): number {
    return this.blankId;
  }

  tokenIdsToText(tokenIds: readonly number[]): string {
    const joined = tokenIds
      .filter((id) => id !== this.blankId)
      .map((id) => this.vocab.get(id) ?? "")
      .filter((token) => token && token !== "<unk>" && token !== "<blank>")
      .join("")
      .replaceAll(WORD_PREFIX, " ");
    return normalizeArabic(joined).trim();
  }

  tokenIdsToRawTokens(tokenIds: readonly number[]): string[] {
    return tokenIds
      .filter((id) => id !== this.blankId)
      .map((id) => this.vocab.get(id) ?? "")
      .filter((token) => token && token !== "<unk>");
  }

  tokenIdsToWordEnds(tokenIds: readonly number[]): number[] {
    const tokens = this.tokenIdsToRawTokens(tokenIds);
    const ends: number[] = [];
    let inWord = false;
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      if (token === WORD_PREFIX || token.startsWith(WORD_PREFIX)) {
        if (inWord) ends.push(i);
        inWord = token !== WORD_PREFIX;
        continue;
      }
      inWord = true;
    }
    if (inWord) ends.push(tokens.length);
    return ends.filter((end, idx) => idx === 0 || end > ends[idx - 1]);
  }
}
