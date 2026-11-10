import { collapseHeld, subCost, UNIT } from "./fuzzy.js";

/** Complete pronunciation identity under existing soft substitutions; no missing or surplus phone. */
export function completePronunciationMatch(written: string, heard: string): boolean {
  const clean = (s: string) => {
    const p = [...s].filter(c => !"ؙۜڇ".includes(c));
    const edge = new Set("ۦۥں۾");
    while (p.length && edge.has(p[0])) p.shift();
    while (p.length && edge.has(p[p.length-1])) p.pop();
    return collapseHeld(p);
  };
  const a = clean(written), b = clean(heard);
  return a.length > 0 && a.length === b.length && a.every((p,i) => subCost(p,b[i]) < UNIT);
}
