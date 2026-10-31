import type { DecodedPhoneme } from "@tilawa/core";

const add = (a: number, b: number) => a === -Infinity ? b : b === -Infinity ? a : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
type Beam = { labels: number[]; b: number; n: number; vb: number; vn: number; tb: DecodedPhoneme[]; tn: DecodedPhoneme[] };
const empty = (labels: number[]): Beam => ({ labels, b: -Infinity, n: -Infinity, vb: -Infinity, vn: -Infinity, tb: [], tn: [] });
const trace = (b: Beam) => b.vb >= b.vn ? b.tb : b.tn;
const score = (b: Beam) => add(b.b, b.n);

/** Bounded, language-model-free CTC prefix search. A one-second lag bounds ambiguous prefixes. */
export function createPrefixDecoder(tokens: string, width = 8, lagFrames = 25) {
  let first = empty([]); first.b = first.vb = 0;
  let beams = [first];
  const pending: DecodedPhoneme[] = [];

  function push(row: readonly (readonly [number, number])[], frame: number) {
    const next = new Map<string, Beam>();
    const put = (labels: number[], blank: boolean, mass: number, v: number, phones: DecodedPhoneme[]) => {
      if (mass === -Infinity) return;
      const key = labels.join(",");
      let b = next.get(key);
      if (!b) { b = empty(labels); next.set(key, b); }
      if (blank) {
        b.b = add(b.b, mass);
        if (v > b.vb) { b.vb = v; b.tb = phones; }
      } else {
        b.n = add(b.n, mass);
        if (v > b.vn) { b.vn = v; b.tn = phones; }
      }
    };
    for (const b of beams) for (const [id, p] of row) {
      if (!(p > 0)) continue;
      const lp = Math.log(p), best = Math.max(b.vb, b.vn);
      if (!id) put(b.labels, true, score(b) + lp, best + lp, trace(b));
      else if (id === b.labels.at(-1)) {
        if (b.n !== -Infinity) {
          const tail = b.tn.at(-1)!;
          put(b.labels, false, b.n + lp, b.vn + lp, [...b.tn.slice(0, -1), { ...tail, end: frame, conf: Math.max(tail.conf, p) }]);
        }
        put([...b.labels, id], false, b.b + lp, b.vb + lp, [...b.tb, { p: tokens[id - 1], start: frame, end: frame, conf: p }]);
      } else put([...b.labels, id], false, score(b) + lp, best + lp, [...trace(b), { p: tokens[id - 1], start: frame, end: frame, conf: p }]);
    }
    beams = [...next.values()].sort((a, b) => score(b) - score(a)).slice(0, width);
    if (!beams.length) throw new Error("CTC frame has no positive probability");
    let common = 0;
    while (common < beams[0].labels.length && beams.every(b => b.labels[common] === beams[0].labels[common])) common++;
    const bestTrace = trace(beams[0]);
    // Never emit the last unresolved label: a following nonblank frame may extend its run.
    let count = Math.max(0, common - 1);
    while (count < bestTrace.length - 1 && bestTrace[count].end <= frame - lagFrames) count++;
    if (count) {
      const prefix = beams[0].labels.slice(0, count);
      pending.push(...bestTrace.slice(0, count));
      beams = beams.filter(b => prefix.every((id, i) => b.labels[i] === id)).map(b => ({ ...b,
        labels: b.labels.slice(count), tb: b.tb.slice(count), tn: b.tn.slice(count) }));
    }
  }
  return {
    push,
    take(): DecodedPhoneme[] { return pending.splice(0); },
    end(): DecodedPhoneme[] { pending.push(...trace(beams[0])); beams = [empty([])]; return pending.splice(0); },
  };
}
