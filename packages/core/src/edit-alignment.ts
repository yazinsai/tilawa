import { align } from "./align.js";

/** A normalized generative edit channel, fitted only to reconstructed training speech. */
export interface EditAlignmentModel {
  phones: string[];
  /** At a reference cursor, insert and stay, or advance the reference. */
  insertion: number;
  inserted: number[];
  /** On advancing, delete or emit a phone drawn from the corresponding row. */
  deleted: number[];
  emitted: number[][];
}

/** Jeffreys pseudo-count, fixed before looking at any practice path. */
const ALPHA = 0.5;

export function fitEditAlignment(phones: readonly string[], pairs: readonly { written: string; heard: string }[]): EditAlignmentModel {
  if (!phones.length || new Set(phones).size !== phones.length || !pairs.length) throw new Error("Empty or invalid alignment supervision");
  const ids = new Map(phones.map((p, i) => [p, i]));
  const id = (p: string) => {
    const i = ids.get(p);
    if (i === undefined) throw new Error(`Training phone outside inventory: ${p}`);
    return i;
  };
  const inserted = Array(phones.length).fill(ALPHA) as number[];
  const deleted = Array(phones.length).fill(ALPHA) as number[];
  const emitted = phones.map(() => Array(phones.length).fill(ALPHA) as number[]);
  let nInsert = ALPHA, nAdvance = ALPHA;
  for (const pair of pairs) {
    if (!pair.written) throw new Error("Training reference is empty");
    for (const op of align(pair.written, pair.heard)) {
      if (op.op === "ins") { inserted[id(op.obs)]++; nInsert++; }
      else {
        nAdvance++;
        if (op.op === "del") deleted[id(op.ref)]++;
        else emitted[id(op.ref)][id(op.op === "match" ? op.ref : op.obs)]++;
      }
    }
    // A finite utterance terminates the terminal cursor's insertion loop.
    nAdvance++;
  }
  const insTotal = inserted.reduce((a, b) => a + b, 0);
  const deletionProbability = deleted.map((count, i) => count / (count + emitted[i].reduce((a, b) => a + b, 0)));
  return {
    phones: [...phones], insertion: nInsert / (nInsert + nAdvance),
    inserted: inserted.map(n => n / insTotal), deleted: deletionProbability,
    emitted: emitted.map(row => { const total = row.reduce((a, b) => a + b, 0); return row.map(n => n / total); }),
  };
}

/** Log addition is stable even for thousands of low-likelihood tokens. */
export function logAdd(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const hi = Math.max(a, b);
  return hi + Math.log1p(Math.exp(Math.min(a, b) - hi));
}

/** Negative-log edit probabilities in the tracker's existing units (three nats per edit). */
export function editAlignmentCosts(model: EditAlignmentModel) {
  const ids = new Map(model.phones.map((p, i) => [p, i]));
  const scale = 10 / 3;
  const id = (p: string) => {
    const i = ids.get(p);
    if (i === undefined) throw new Error(`Alignment phone outside inventory: ${p}`);
    return i;
  };
  const advance = 1 - model.insertion;
  return {
    insertion: (p: string) => -Math.log(model.insertion * model.inserted[id(p)]) * scale,
    deletion: (p: string) => -Math.log(advance * model.deleted[id(p)]) * scale,
    emission: (r: string, p: string) => -Math.log(advance * (1 - model.deleted[id(r)]) * model.emitted[id(r)][id(p)]) * scale,
    sum: (a: number, b: number) => -logAdd(-a / scale, -b / scale) * scale,
  };
}

/**
 * P(heard | written), summing every token edit alignment. Insertions, deletions and
 * emissions compete on a probability simplex; adding routes cannot create free mass.
 * The optional max reduction is a diagnostic using exactly the same frozen model.
 */
export function editAlignmentLogP(written: string, heard: string, model: EditAlignmentModel, reduction: "sum" | "max" = "sum"): number {
  const ids = new Map(model.phones.map((p, i) => [p, i]));
  const id = (p: string) => {
    const i = ids.get(p);
    if (i === undefined) throw new Error(`Alignment phone outside inventory: ${p}`);
    return i;
  };
  const wi = [...written].map(id), hi = [...heard].map(id);
  const add = reduction === "sum" ? logAdd : Math.max;
  const logAdvance = Math.log1p(-model.insertion);
  const ins = hi.map(i => Math.log(model.insertion * model.inserted[i]));
  const row = new Float64Array(heard.length + 1);
  for (let j = 1; j < row.length; j++) row[j] = row[j - 1] + ins[j - 1];
  for (const w of wi) {
    const del = logAdvance + Math.log(model.deleted[w]);
    const emit = logAdvance + Math.log1p(-model.deleted[w]);
    let diag = row[0];
    row[0] += del;
    for (let j = 1; j < row.length; j++) {
      const prev = row[j];
      row[j] = add(add(prev + del, row[j - 1] + ins[j - 1]), diag + emit + Math.log(model.emitted[w][hi[j - 1]]));
      diag = prev;
    }
  }
  return row[heard.length] + logAdvance;
}
