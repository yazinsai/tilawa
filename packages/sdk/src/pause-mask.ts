import { minGapFrames, waqfMarked } from "@tilawa/core";
import type { PhonemeEvidence, ReferenceIndex } from "@tilawa/core";
import type { LockPoint } from "./gap-decode.js";

export interface PauseSpan { fromFrame: number; toFrame: number }

/** The existing pause-trained ear's detector/step mask, evaluated on its own full run. */
export function pauseMaskSpans(chunks: PhonemeEvidence[], locks: LockPoint[], index: ReferenceIndex): PauseSpan[] {
  if (!chunks.length || chunks.some(c => !c.posteriors)) return [];
  const frameMs = chunks[0].frameMs;
  const rows = chunks.flatMap(c => {
    const p = c.posteriors!;
    return Array.from({length:p.nFrames},(_,t) => ({ frame:p.startFrame+t,
      blank:Array.from({length:p.k},(_,j)=>j).some(j=>p.ids[t*p.k+j]===0 && p.probs[t*p.k+j]>=.5) }));
  });
  const pauses: PauseSpan[] = [];
  for (let a=0;a<rows.length;) {
    let b=a+1;
    while(b<rows.length && rows[b].frame===rows[b-1].frame+1 && rows[b].blank===rows[a].blank) b++;
    if(rows[a].blank && b-a>=minGapFrames(frameMs)) pauses.push({fromFrame:rows[a].frame,toFrame:rows[b-1].frame+1});
    a=b;
  }
  const nextWord=(wi:number)=>{do{wi++;}while(wi<index.words.length && index.words[wi].end===index.words[wi].start);return wi;};
  const spans: PauseSpan[]=[];
  for(let i=1;i+1<locks.length;i++) {
    const left=locks[i-1],first=locks[i],second=locks[i+1];
    if(first.wi!==nextWord(left.wi)||second.wi!==nextWord(first.wi)) continue;
    const prev=index.words[left.wi],word=index.words[first.wi];
    if(!prev||!word||prev.surah!==word.surah) continue;
    if(!(prev.ayah!==word.ayah && word.word===1) && !(prev.ayah===word.ayah && waqfMarked(left.wi))) continue;
    const end=Math.ceil(second.timeMs/frameMs);
    for(const p of pauses.filter(p=>p.fromFrame>=left.timeMs/frameMs && p.fromFrame<=first.timeMs/frameMs && p.toFrame<end))
      spans.push({fromFrame:p.fromFrame,toFrame:end});
  }
  return spans;
}

/** Original mask clock: two 20-ms embedding frames per 40-ms CTC frame. */
export function pauseFloor(spans: PauseSpan[], processed: number, chunk: number): BigInt64Array {
  const floors=new BigInt64Array(chunk).fill(-1n);
  for(let q=0;q<chunk;q++) {
    const frame=(processed+q)/2;
    const active=spans.filter(s=>frame>=s.fromFrame && frame<s.toFrame).at(-1);
    if(active) floors[q]=BigInt(active.fromFrame*2);
  }
  return floors;
}
