export interface CTCResult {
  /** Phonemes within words concatenated, words space-separated: "bismi allahi" */
  text: string;
  /** Raw phoneme tokens space-separated: "b i s m i | a l l a h i" */
  rawPhonemes: string;
  tokenIds: number[];
}

export class CTCDecoder {
  private vocab: Map<number, string>;
  private tokenToId: Map<string, number>;
  private blankId: number;

  constructor(vocabJson: Record<string, string>) {
    this.vocab = new Map();
    this.tokenToId = new Map();
    this.blankId = -1;
    for (const [id, token] of Object.entries(vocabJson)) {
      const numId = parseInt(id);
      this.vocab.set(numId, token);
      this.tokenToId.set(token, numId);
      if (token === "<blank>") {
        this.blankId = numId;
      }
    }
    // Fallback: blank is last token if not found by value
    if (this.blankId === -1) {
      let maxId = 0;
      for (const id of this.vocab.keys()) {
        if (id > maxId) maxId = id;
      }
      this.blankId = maxId;
    }
  }

  decode(logprobs: Float32Array, timeSteps: number, vocabSize: number): CTCResult {
    // argmax per timestep
    const ids: number[] = [];
    for (let t = 0; t < timeSteps; t++) {
      let maxIdx = 0;
      let maxVal = logprobs[t * vocabSize];
      for (let v = 1; v < vocabSize; v++) {
        const val = logprobs[t * vocabSize + v];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = v;
        }
      }
      ids.push(maxIdx);
    }

    // Collapse consecutive duplicates, remove blanks
    const tokens: string[] = [];
    let prev = -1;
    for (const id of ids) {
      if (id !== prev && id !== this.blankId) {
        const token = this.vocab.get(id) ?? "";
        tokens.push(token);
      }
      prev = id;
    }

    // Raw phonemes: all tokens space-separated
    const rawPhonemes = tokens.join(" ");

    // Joined text: concatenate within words, split on |
    const words: string[] = [];
    let currentWord: string[] = [];
    for (const tok of tokens) {
      if (tok === "|") {
        if (currentWord.length > 0) {
          words.push(currentWord.join(""));
        }
        currentWord = [];
      } else {
        currentWord.push(tok);
      }
    }
    if (currentWord.length > 0) {
      words.push(currentWord.join(""));
    }

    return {
      text: words.join(" "),
      rawPhonemes,
      tokenIds: tokens
        .map((token) => this.tokenToId.get(token) ?? -1)
        .filter((id) => id >= 0),
    };
  }

  getBlankId(): number {
    return this.blankId;
  }

  encodeRawPhonemes(rawPhonemes: string): number[] {
    const ids: number[] = [];
    for (const token of rawPhonemes.trim().split(/\s+/)) {
      if (!token) continue;
      const id = this.tokenToId.get(token);
      if (id !== undefined && id !== this.blankId) {
        ids.push(id);
      }
    }
    return ids;
  }
}
