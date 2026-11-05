import { UNIT, subCost } from "./fuzzy.js";

/** Existing checker charges: consonants are hard, vowels and recitation marks soft, repeated sounds held. */
export const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");
export const SOFT = 3;
export const HELD = 2;

export const subCharge = (r: string, p: string) =>
  r === p ? 0 : CONSONANTS.has(r) && CONSONANTS.has(p) ? subCost(r, p) : SOFT;
export const delCharge = (ref: string, at: number) =>
  ref[at] === ref[at - 1] || ref[at] === ref[at + 1] ? HELD : CONSONANTS.has(ref[at]) ? UNIT : SOFT;
export const insCharge = (ref: string, at: number, p: string, prev?: string) =>
  p === ref[at - 1] || p === ref[at] || p === prev ? HELD : CONSONANTS.has(p) ? UNIT : SOFT;

/** Full-word minimum-charge fit. Equal charges use the fewest hard edits. No free prefixes or suffixes. */
export function wordFit(ref: string, heard: string): { cost: number; hard: number } {
  const width = heard.length + 1;
  const costs = new Float64Array((ref.length + 1) * width);
  const hard = new Uint32Array(costs.length);
  const put = (cell: number, before: number, charge: number) => {
    const c = costs[before] + charge, h = hard[before] + Number(charge >= UNIT);
    if (c < costs[cell] || c === costs[cell] && h < hard[cell]) { costs[cell] = c; hard[cell] = h; }
  };
  for (let i = 0; i <= ref.length; i++) for (let j = 0; j <= heard.length; j++) {
    const cell = i * width + j;
    if (!cell) continue;
    costs[cell] = Infinity;
    if (i && j) put(cell, cell - width - 1, subCharge(ref[i - 1], heard[j - 1]));
    if (i) put(cell, cell - width, delCharge(ref, i - 1));
    if (j) put(cell, cell - 1, insCharge(ref, i, heard[j - 1], heard[j - 2]));
  }
  return { cost: costs[costs.length - 1], hard: hard[hard.length - 1] };
}
