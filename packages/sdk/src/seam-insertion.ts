import { createChecker, createTracker, trackerInput, PosteriorFrames, PHRASE_SWITCH, seamInsertionMarginal, seamPhoneLoop,
  shadowInsertionIntervals, minGapFrames, collapseHeld, lexicalWords } from "@tilawa/core";
import type { DecodedPhoneme, PhonemeEvidence, PhraseAsset, ReferenceIndex, SeamInsertionCandidate, SeamInsertionWindow, SeamInsertionEvidence } from "@tilawa/core";
import type { LoadedModel } from "./model.js";

/** One native frame at a time, extending a cell through a token's entire run.
 * This is an engine evidence clock; no scoring code or labels participate. */
export function insertionCells(chunks: PhonemeEvidence[]): PhonemeEvidence[] {
  if (!chunks.length || chunks.some(c => !c.posteriors)) return [];
  const phones = chunks.flatMap(c => c.phonemes), k = chunks[0].posteriors!.k;
  const ids = chunks.flatMap(c => Array.from(c.posteriors!.ids)), probs = chunks.flatMap(c => Array.from(c.posteriors!.probs));
  const first = chunks[0].posteriors!.startFrame, end = first + ids.length / k;
  const out: PhonemeEvidence[] = [];
  let p = 0;
  for (let a = first; a < end;) {
    let b = a + 1; const phonemes = [];
    while (p < phones.length && phones[p].start < b) { phonemes.push(phones[p]); b = Math.max(b, phones[p++].end + 1); }
    b = Math.min(b, end);
    out.push({ inventory: chunks[0].inventory, frameMs: chunks[0].frameMs, phonemes, decodedTo: b,
      posteriors: { startFrame: a, nFrames: b - a, k, ids: ids.slice((a-first)*k,(b-first)*k), probs: probs.slice((a-first)*k,(b-first)*k) } });
    a = b;
  }
  return out;
}

/** Auxiliary metadata is carried by the first primary token, with all primary values intact. */
export function carryInsertionEvidence(chunks: PhonemeEvidence[], metadata: SeamInsertionEvidence[]): PhonemeEvidence[] {
  if (!metadata.length) return chunks;
  const at = chunks.findIndex(c => c.phonemes.length);
  return chunks.map((c,i) => i !== at ? c : { ...c, phonemes: c.phonemes.map((p,j) => j ? p : { ...p, seamInsertions: metadata }) });
}

/** Read-only candidate discovery and warm teacher judgment. The primary tracker consumes only primary tokens. */
export async function addInsertionEvidence(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset): Promise<PhonemeEvidence[]> {
  if (!model.insertionTeacher || !audio.length) return chunks;
  if (model.manifest.sampleRate !== 16000) throw new Error("Insertion teacher requires 16 kHz audio");
  const cells = insertionCells(chunks);
  if (!cells.length) return chunks;
  const candidates: SeamInsertionCandidate[] = [];
  const seams = new Map<string, SeamInsertionWindow>();
  const steps: { word: number; frame: number }[] = [];
  const wordKey = (w: { surah:number; ayah:number; word:number }) => `${w.surah}:${w.ayah}:${w.word}`;
  const wordIds = new Map(index.words.map((w,i)=>[wordKey(w),i]));
  const seen = new Set<string>();
  const tracker = createTracker(index, { mode: "free", phrases });
  const checker = createChecker(index, { phrases, seamInsertion(c) {
    const key = JSON.stringify(c);
    if (!seen.has(key)) { seen.add(key); candidates.push(c); }
    // Discovery runs the unchanged generator, including its issue state.
    return true;
  }, seamPhoneLoop(c) {
    const key = `${c.word}:${c.prefixFrame ?? c.seamFromFrame}`;
    const prior = seams.get(key);
    if (!prior || c.seamToFrame > prior.seamToFrame) seams.set(key,c);
  } });
  for (const e of cells) {
    const events = tracker.push(trackerInput(e));
    for (const event of events) if (event.type === "position" && event.state === "locked" && event.observed.length) {
      const word = wordIds.get(wordKey(event.observed[0].word));
      if (word !== undefined) steps.push({word,frame:Math.ceil(event.timeMs/e.frameMs)+1});
    }
    checker.push(e,events);
  }
  checker.end(tracker.end());
  const lex = lexicalWords(index);
  const frameMs = cells[0].frameMs, ids = new Map([...index.tokens].map((p,i) => [p,i+1]));
  const forms = (wi: number) => [...new Set([index.phonemes.slice(index.words[wi].start,index.words[wi].end),
    index.isolatedWordForms?.[wi] ?? ""].filter(Boolean).map(collapse))];
  const cache = new Map<string, { frames: PosteriorFrames; shift: number; phones: DecodedPhoneme[] }>();
  const metadata: SeamInsertionEvidence[] = [];
  for (const c of candidates) {
    const prev = previousWord(index,c.word), prev2 = previousWord(index,prev), next = followingWord(index,c.word);
    if (prev < 0 || next >= index.words.length) continue;
    const before = prev2 >= 0 && index.words[prev2].ayah === index.words[prev].ayah &&
      index.words[prev2].surah === index.words[prev].surah ? [prev2,prev] : [prev];
    // A whole two-word prefix needs its actual onset; no tight candidate-frame fallback.
    if (before.length === 2 && c.prefixFrame === undefined) continue;
    const from = c.prefixFrame ?? c.seamFromFrame;
    const start = Math.max(0, from - Math.round(4000/frameMs));
    const end = c.seamToFrame + Math.round(300/frameMs), key = `${start}:${end}`;
    let crop = cache.get(key);
    if (!crop) {
      const offset = Math.round(start*frameMs*16), limit = Math.min(audio.length,Math.round(end*frameMs*16));
      if (limit <= offset) continue;
      const e = await model.insertionTeacher.decode(audio.subarray(offset,limit),index.tokens.size+1);
      if (e.inventory !== index.inventoryId || e.frameMs !== frameMs || !e.posteriors) throw new Error("Insertion teacher clock or inventory differs from primary evidence");
      const frames = new PosteriorFrames(index.tokens.size);
      frames.append({ ...e.posteriors, startFrame: e.posteriors.startFrame + start });
      crop = { frames, shift: start, phones:e.phonemes.map(p=>({...p,start:p.start+start,end:p.end+start})) }; cache.set(key,crop);
    }
    const map = (ws: number[]) => ws.map(w => forms(w).map(f => [...f].map(p => ids.get(p)!)));
    // The candidate's exact soft-only spelling is already certified by the generator.
    const spelling = lex.surplusWord(c.word, c.said);
    if (!spelling) continue;
    const candidateForms = new Set([spelling]);
    const norm = (s: string) => collapseHeld([...s].filter(p => !"ڇؙۜ".includes(p))).join("");
    for (const list of index.wordForms ?? []) for (const f of list)
      if (norm(f) === spelling) candidateForms.add(collapse(f));
    const candidate = [...candidateForms].map(f => [...f].map(p => ids.get(p)!));
    const marginal = seamInsertionMarginal(crop.frames,map(before),candidate,map([c.word,next]),from,
      Math.min(end,crop.frames.endFrame),PHRASE_SWITCH);
    if (marginal !== null) metadata.push({ word: { surah:index.words[c.word].surah, ayah:index.words[c.word].ayah, word:index.words[c.word].word },
      observed:c.said,fromFrame:c.fromFrame,toFrame:c.toFrame,marginal });
  }
  if (!seams.size) return carryInsertionEvidence(chunks,metadata);
  const primary = new PosteriorFrames(index.tokens.size);
  for (const cell of cells) primary.append(cell.posteriors!);
  const loopPhones = [...ids].filter(([p])=>!"ڇؙۜ".includes(p)).map(([,id])=>id);
  const consonants = [...ids].filter(([p])=>"ءبتثجحخدذرزسشصضطظعغفقكلمنهوي".includes(p)).map(([,id])=>id);
  const inventory = [...index.tokens];
  for (const c of seams.values()) {
    const prev=previousWord(index,c.word), prev2=previousWord(index,prev), next=followingWord(index,c.word);
    if (prev<0 || next>=index.words.length) continue;
    const before=prev2>=0 && index.words[prev2].ayah===index.words[prev].ayah &&
      index.words[prev2].surah===index.words[prev].surah ? [prev2,prev] : [prev];
    if (before.length===2 && c.prefixFrame===undefined) continue;
    const from=c.prefixFrame??c.seamFromFrame, start=Math.max(0,from-Math.round(4000/frameMs));
    const end=c.seamToFrame+Math.round(300/frameMs), key=`${start}:${end}`;
    let crop=cache.get(key);
    if (!crop) {
      const offset=Math.round(start*frameMs*16),limit=Math.min(audio.length,Math.round(end*frameMs*16));
      if (limit<=offset) continue;
      const e=await model.insertionTeacher.decode(audio.subarray(offset,limit),index.tokens.size+1);
      if (e.inventory!==index.inventoryId || e.frameMs!==frameMs || !e.posteriors)
        throw new Error("Insertion teacher clock or inventory differs from primary evidence");
      const frames=new PosteriorFrames(index.tokens.size);
      frames.append({...e.posteriors,startFrame:e.posteriors.startFrame+start});
      crop={frames,shift:start,phones:e.phonemes.map(p=>({...p,start:p.start+start,end:p.end+start}))};
      cache.set(key,crop);
    }
    const intervals=new Map<string, readonly [number,number]>();
    for (let i=0;i<steps.length;i++) {
      if (steps[i].word!==c.word || steps[i].frame<c.seamFromFrame || steps[i].frame>c.seamToFrame) continue;
      const step=steps.slice(i+1).find(s=>s.word===next && s.frame<=c.seamToFrame);
      if (!step) continue;
      const bound=Math.min(step.frame,primary.endFrame), a=Math.max(from,primary.startFrame);
      let pause=-1;
      for (let f=a;f<=bound;f++) {
        const blank=f<bound && primary.row(f)[0]>=Math.log(.5);
        if (blank && pause<0) pause=f;
        if (!blank && pause>=0) {
          if (f-pause>=minGapFrames(frameMs)) {
            // Student blanks may cover teacher speech: retain teacher tokens from the pause's start.
            const phones=crop.phones.filter(p=>p.start>=pause && p.start<bound)
              .map(p=>({...p,end:Math.min(p.end,bound-1)}));
            for (const fit of shadowInsertionIntervals(index,c.word,phones,frameMs))
              if (fit.fromFrame>=c.seamFromFrame) intervals.set(`${fit.fromFrame}:${fit.toFrame}`,[fit.fromFrame,fit.toFrame]);
          }
          pause=-1;
        }
      }
    }
    const map=(ws:number[])=>ws.map(w=>forms(w).map(f=>[...f].map(p=>ids.get(p)!)));
    const to=Math.min(end,crop.frames.endFrame);
    for (const interval of intervals.values()) {
      const result=seamPhoneLoop(crop.frames,map(before),map([c.word,next]),loopPhones,consonants,from,to,PHRASE_SWITCH,interval);
      if (!result || result.marginal<=.5 || !result.phones.length) continue;
      const observed=result.phones.map(p=>inventory[p-1]).join("");
      if (lex.fragmentOfSeam(c.word,observed)) continue;
      metadata.push({kind:"phone-loop",word:{surah:index.words[c.word].surah,ayah:index.words[c.word].ayah,word:index.words[c.word].word},
        observed,fromFrame:result.fromFrame,toFrame:result.toFrame,marginal:result.marginal});
    }
  }
  return carryInsertionEvidence(chunks,metadata);
}

// Same held-sound collapse as the core's spelling comparison; ordinary doubled consonants stay doubled.
function collapse(s: string): string {
  return collapseHeld([...s]).join("");
}
function previousWord(index: ReferenceIndex, at: number): number { do { at--; } while (at >= 0 && index.words[at].end === index.words[at].start); return at; }
function followingWord(index: ReferenceIndex, at: number): number { do { at++; } while (at < index.words.length && index.words[at].end === index.words[at].start); return at; }
