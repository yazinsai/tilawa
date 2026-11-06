import type { EngineEvent } from "./events.js";
import type { PhonemeEvidence } from "./evidence.js";
import type { ReferenceIndex } from "./reference.js";
import { collapseHeld } from "./fuzzy.js";
import { waqfMarked } from "./mushaf-marks.js";

type Arc = { src: number; dst: number };
type WordCountFrames = { startFrame: number; endFrame: number; row(frame:number): ArrayLike<number> };
export interface WordCountGraph {
  labels: number[];
  arcs: Arc[];
  finals: number[];
  slot: number[];
}

/** Closed separator CTC choices. The extra slot is entered once, bounded by
 * separators, and must emit at least two CTC consonants. A held class counts
 * once; a second copy needs an intervening blank. Repeats retain separators. */
export function wordCountGraph(words: readonly (readonly (readonly number[])[])[],
  separator: number, phones: readonly number[], consonants: ReadonlySet<number>, extra: boolean,
  repeats = true): WordCountGraph {
  if (words.length !== 3 || !Number.isInteger(separator) || separator <= 0 ||
    words.some(w => !w.length || w.some(f => !f.length || f.some(p => p <= 0 || p === separator))) ||
    phones.some(p => p <= 0 || p === separator) || !phones.some(p => consonants.has(p)))
    throw new Error("Invalid separator word-count alphabet or text");
  const labels = [0], arcs: Arc[] = [], slot: number[] = [], seen = new Set<string>();
  const state = (label: number, inSlot = false) => {
    labels.push(label); const s = labels.length - 1;
    if (inSlot) slot.push(s);
    return s;
  };
  const arc = (src: number, dst: number) => {
    const key = `${src}:${dst}`;
    if (!seen.has(key)) { seen.add(key); arcs.push({ src, dst }); }
  };
  const connect = (sources: readonly number[], destinations: readonly number[], repeat = false) => {
    for (const src of sources) for (const dst of destinations)
      if (src !== dst && (repeat || !labels[src] || labels[src] !== labels[dst])) arc(src, dst);
  };
  const boundary = (sources: number[], inSlot = false) => {
    const s = state(separator, inSlot), b = state(0, inSlot);
    connect(sources, [s]); arc(s, b);
    return [s, b];
  };
  const word = (forms: readonly (readonly number[])[], sources: number[]) => {
    const first: number[] = [], last: number[] = [], b = state(0);
    for (const f of new Map(forms.map(f => [JSON.stringify(f), f])).values()) {
      let prev = -1;
      for (const p of f) {
        const s = state(p);
        if (prev < 0) first.push(s);
        else { const blank = state(0); arc(prev, blank); arc(blank, s); connect([prev], [s]); }
        prev = s;
      }
      last.push(prev); arc(prev, b);
    }
    connect(sources, first);
    if (repeats) connect(boundary([...last, b]), first);
    return { first, tail: [...last, b] };
  };
  const left = word(words[0], [0]);
  let seam = boundary(left.tail);
  if (extra) {
    // The seam's blank already owns pre-slot silence. No second empty-slot
    // blank state: that would count identical timing paths twice.
    const blanks = [-1, state(0, true), state(0, true)];
    const emits = Array.from({ length: 3 }, () => new Map<number, number>());
    for (let count = 0; count <= 2; count++) for (const p of new Set(phones)) {
      if (!count && consonants.has(p)) continue;
      emits[count].set(p, state(p, true));
    }
    for (const p of new Set(phones)) connect(seam, [emits[consonants.has(p) ? 1 : 0].get(p)!]);
    for (let count = 0; count <= 2; count++) for (const [p, s] of emits[count]) {
      if (count) arc(s, blanks[count]);
      else {
        // Zero consonants can still contain vowels. A private blank after
        // those vowels is reachable only after speech, never from the seam.
        if (blanks[0] < 0) blanks[0] = state(0, true);
        arc(s, blanks[0]);
      }
      for (const q of new Set(phones)) {
        const next = Math.min(2, count + Number(consonants.has(q)));
        if (p !== q) arc(s, emits[next].get(q)!);
      }
    }
    for (let count = 0; count <= 2; count++) if (blanks[count] >= 0)
      for (const p of new Set(phones)) arc(blanks[count], emits[Math.min(2, count + Number(consonants.has(p)))].get(p)!);
    seam = boundary([...emits[2].values(), blanks[2]]);
  }
  const right = word(words[1], seam);
  const last = word(words[2], boundary(right.tail));
  if (repeats) connect(boundary(last.tail), right.first);
  for (let s = 0; s < labels.length; s++) arc(s, s);
  return { labels, arcs, finals: last.tail, slot };
}

/** Forward sum over all frame timings; primary evidence is never mutated. */
export function wordCountLogSum(graph: WordCountGraph,
  frames: WordCountFrames, from: number, to: number): number | null {
  if (from < frames.startFrame || to > frames.endFrame || to <= from) return null;
  let prev = new Float64Array(graph.labels.length); prev[0] = 1;
  let scale = 0;
  for (let f = from; f < to; f++) {
    const row = frames.row(f), next = new Float64Array(prev.length);
    if (graph.labels.some(p => p >= row.length)) return null;
    for (const a of graph.arcs) next[a.dst] += prev[a.src];
    let peak = 0;
    for (let s = 0; s < next.length; s++) {
      next[s] *= Math.exp(row[graph.labels[s]]);
      peak = Math.max(peak,next[s]);
    }
    if (!peak) return null;
    // Scale every frame, so long blank pauses cannot underflow the sum.
    for (let s = 0; s < next.length; s++) next[s] /= peak;
    scale += Math.log(peak);
    prev = next;
  }
  let sum = 0;
  for (const s of graph.finals) sum += prev[s];
  return sum > 0 ? Math.log(sum) + scale : null;
}

/** Best complete path in the extra-count choice, retaining only its slot.
 * The caller compares forward sums; a different best no-slot timing cannot
 * erase the phones of an accepted extra-count hypothesis. */
export function wordCountSlot(graph: WordCountGraph,
  frames: WordCountFrames, from: number, to: number):
  { phones: number[]; fromFrame: number; toFrame: number } | null {
  if (!graph.slot.length || from < frames.startFrame || to > frames.endFrame || to <= from) return null;
  const S = graph.labels.length, trace: Int32Array[] = [];
  let prev = new Float64Array(S).fill(-Infinity); prev[0] = 0;
  for (let frame = from; frame < to; frame++) {
    const row = frames.row(frame), next = new Float64Array(S).fill(-Infinity);
    if (graph.labels.some(p => p >= row.length)) return null;
    const back = new Int32Array(S).fill(-1);
    for (const arc of graph.arcs) {
      const score = prev[arc.src] + row[graph.labels[arc.dst]];
      if (score > next[arc.dst]) { next[arc.dst] = score; back[arc.dst] = arc.src; }
    }
    trace.push(back); prev = next;
  }
  let state = graph.finals.reduce((best,s) => prev[s] > prev[best] ? s : best, graph.finals[0]);
  if (!Number.isFinite(prev[state])) return null;
  const path = new Int32Array(to-from);
  for (let t = path.length-1; t >= 0; t--) { path[t] = state; state = trace[t][state]; }
  const slot = new Set(graph.slot), phones: number[] = [];
  let start = -1, end = -1, last = 0;
  for (let t = 0; t < path.length; t++) {
    const p = graph.labels[path[t]];
    if (slot.has(path[t]) && p) {
      if (start < 0) start = from+t;
      end = from+t+1;
      if (p !== last) phones.push(p);
    }
    last = p;
  }
  return start >= 0 ? { phones, fromFrame:start, toFrame:end } : null;
}

export interface WordCountAddition {
  word: number;
  observed: string;
  fromFrame: number;
  toFrame: number;
  margin: number;
}

/** Read-only shadow of primary tracker steps. Rows are never decoded into the
 * main string, and no label, text identity for the slot, or visit ownership
 * participates in the count contest. */
export function createWordCountDetector(index: ReferenceIndex) {
  const key = (w: {surah:number;ayah:number;word:number}) => `${w.surah}:${w.ayah}:${w.word}`;
  const wi = new Map(index.words.map((w,i) => [key(w),i]));
  const rows = new Map<number, Float32Array>();
  const steps = new Map<number,number>();
  let tokens = "", cleanMax = Infinity, last = -1;
  let ids = new Map<string,number>(), separator = -1, phones: number[] = [], consonants = new Set<number>();
  const variants = (at:number) => [...new Set([...(index.wordForms?.[at] ?? []),
    index.phonemes.slice(index.words[at].start,index.words[at].end),index.isolatedWordForms?.[at] ?? ""]
    .filter(Boolean).map(s => collapseHeld([...s]).join("")))].map(s => [...s].map(p => ids.get(p)!));
  return { push(evidence: PhonemeEvidence, events: EngineEvent[]): WordCountAddition[] {
    for (const p of evidence.phonemes) for (const packet of p.wordCountPosteriors ?? []) {
      if (!packet.tokens.includes("|") || !Number.isFinite(packet.cleanMaxMargin) ||
        !Number.isInteger(packet.startFrame) || packet.startFrame < 0 || !Number.isInteger(packet.nFrames) || packet.nFrames <= 0 ||
        packet.logProbs.length !== packet.nFrames*(packet.tokens.length+1) || packet.logProbs.some(v => !Number.isFinite(v))) continue;
      if (tokens && (packet.tokens !== tokens || packet.cleanMaxMargin !== cleanMax)) continue;
      if (!tokens) {
        tokens = packet.tokens; cleanMax = packet.cleanMaxMargin;
        ids = new Map([...tokens].map((p,i) => [p,i+1])); separator = ids.get("|")!;
        phones = [...tokens].flatMap((p,i) => "|ڇؙۜ".includes(p) ? [] : [i+1]);
        consonants = new Set(phones.filter(i => "ءبتثجحخدذرزسشصضطظعغفقكلمنهوي".includes(tokens[i-1])));
      }
      const width = tokens.length+1;
      for (let t=0;t<packet.nFrames;t++) rows.set(packet.startFrame+t,Float32Array.from(packet.logProbs.slice(t*width,(t+1)*width)));
    }
    const additions: WordCountAddition[] = [];
    for (const event of events) {
      if (event.type !== "position") continue;
      if (event.state !== "locked" || !event.expected) {last=-1;steps.clear();continue;}
      const at = wi.get(key(event.expected));
      if (at === undefined || at === last) continue;
      if (last>=0 && at!==last+1) steps.clear();
      last = at; const frame = Math.round(event.timeMs/evidence.frameMs); steps.set(at,frame);
      const prev=at-2,w1=at-1,start=steps.get(prev),middle=steps.get(w1);
      if (tokens && prev>=0 && start!==undefined && middle!==undefined && frame>start &&
        index.words[prev].surah===index.words[w1].surah &&
        (index.words[w1].word===1 || waqfMarked(prev))) {
        let covered = true;
        for(let f=start;f<frame;f++) if(!rows.has(f)) {covered=false;break;}
        const words = [variants(prev),variants(w1),variants(at)];
        if (covered && words.every(w=>w.length && w.every(f=>f.length && f.every(p=>Number.isInteger(p))))) {
          const frames = {startFrame:start,endFrame:frame,row:(f:number)=>rows.get(f)!};
          const no=wordCountLogSum(wordCountGraph(words,separator,phones,consonants,false),frames,start,frame);
          const extra=wordCountGraph(words,separator,phones,consonants,true);
          const yes=wordCountLogSum(extra,frames,start,frame);
          if(no!==null && yes!==null && yes-no>cleanMax) {
            const slot=wordCountSlot(extra,frames,start,frame);
            if(slot) additions.push({word:w1,observed:slot.phones.map(p=>tokens[p-1]).join(""),
              fromFrame:slot.fromFrame,toFrame:slot.toFrame,margin:yes-no});
          }
        }
      }
      // Only the next two written steps can use older head rows.
      const floor=steps.get(at-1) ?? frame;
      for(const f of rows.keys()) if(f<floor) rows.delete(f);
      for(const word of steps.keys()) if(word<at-1) steps.delete(word);
    }
    return additions;
  } };
}
