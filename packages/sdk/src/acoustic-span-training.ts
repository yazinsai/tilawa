/** Training transcript construction never treats a scripted deviation as canonical. */
import type { ReferenceIndex, WordRef } from "@tilawa/core";
export interface PronunciationLabel {
  kind: string; ayahs: string[]; scored?: boolean;
  events?: { errorClass: string; state: string; location: WordRef; observed?: string }[];
  non_errors?: unknown[]; unsupported?: unknown[];
}
export function trainingPronunciations(index: ReferenceIndex, label: PronunciationLabel): {
  kind: "none" | "supported-substitution"; written: string[]; recorded: string[];
} | null {
  if (label.scored === false || label.non_errors?.length || label.unsupported?.length) return null;
  const words=index.words.filter(w=>label.ayahs.includes(`${w.surah}:${w.ayah}`) && w.end>w.start);
  const written=words.map(w=>index.phonemes.slice(w.start,w.end));
  if (!words.length) return null;
  const events=label.events??[];
  if (label.kind==="none") return events.length ? null : {kind:"none",written,recorded:[...written]};
  if (!events.length || events.some(e=>e.errorClass!=="wrong_word" || e.state!=="confirmed" || !e.observed)) return null;
  const recorded=[...written], replaced=new Set<number>();
  for (const e of events) {
    const at=words.findIndex(w=>w.surah===e.location.surah && w.ayah===e.location.ayah && w.word===e.location.word);
    if (at<0 || replaced.has(at)) return null;
    // Only an explicit complete attested scripted spelling is an eligible candidate.
    const forms=new Set<string>();
    index.wordText?.forEach((text,i)=>{
      if (text.normalize("NFC")===e.observed!.normalize("NFC")) {
        const form=index.isolatedWordForms?.[i] ?? index.phonemes.slice(index.words[i].start,index.words[i].end);
        if (form) forms.add(form);
      }
    });
    if (forms.size!==1) return null;
    recorded[at]=[...forms][0]; replaced.add(at);
  }
  if (recorded.join("")===written.join("")) return null;
  return {kind:"supported-substitution",written,recorded};
}
const logadd=(a:number,b:number)=>{
  if(a===-Infinity)return b;if(b===-Infinity)return a;
  const m=Math.max(a,b);return m+Math.log(Math.exp(a-m)+Math.exp(b-m));
};
/** Full CTC path mass is used only to reject unsupported scripted training labels. */
export function transcriptLogProbability(lp:number[][], ids:number[]):number {
  if(!lp.length || !ids.length || ids.some(i=>i<1 || i>=lp[0].length))return -Infinity;
  const labels=[0,...ids.flatMap(p=>[p,0])];
  let a=labels.map(()=>-Infinity);a[0]=lp[0][0];a[1]=lp[0][ids[0]];
  for(let t=1;t<lp.length;t++) a=labels.map((p,s)=>{
    let value=a[s];if(s>0)value=logadd(value,a[s-1]);
    if(s>1 && p!==0 && p!==labels[s-2])value=logadd(value,a[s-2]);
    return value+lp[t][p];
  });
  return logadd(a.at(-1)!,a.at(-2)!);
}
