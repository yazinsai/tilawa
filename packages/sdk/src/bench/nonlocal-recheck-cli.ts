/** Inspect reconstruction on actual string-path checker requests, without scoring or changing evidence.
 * --audio WAV --evidence JSON --model DIR --calibration JSON --out JSON [--word surah:ayah:word] */
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { loadReferenceIndex, completePronunciationMatch } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { loadNodeModel } from "../node.js";
import { decodeWav } from "../wav.js";
import { Resampler } from "../audio.js";
import { NonlocalAcousticDenoiser } from "../nonlocal-acoustic.js";
import { wrongWordRechecks, replayAcousticSlot } from "../nonlocal-recheck.js";

const { values } = parseArgs({options:{audio:{type:"string"},evidence:{type:"string"},model:{type:"string"},calibration:{type:"string"},out:{type:"string"},word:{type:"string"}}});
if (!values.audio || !values.evidence || !values.model || !values.out || !values.calibration)
  throw new Error("--audio, --evidence, --model, --calibration, --out required");
const calibration = JSON.parse(readFileSync(values.calibration,"utf8"));
if (calibration.protocol !== "nonlocal-independent-noise-v1" || calibration.validation?.length !== 3 || !calibration.selected)
  throw new Error("A completed speaker-grouped training calibration is required");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../../core/assets/ref-v2.json",import.meta.url),"utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../../core/assets/phrases-v1.json",import.meta.url),"utf8"));
const model = await loadNodeModel(values.model);
const source = JSON.parse(readFileSync(values.evidence,"utf8"));
const chunks: PhonemeEvidence[] = Array.isArray(source) ? source : [{...source,
  ...(source.posteriors && {posteriors:{...source.posteriors,startFrame:source.posteriors.startFrame ?? 0,
    nFrames:source.posteriors.nFrames ?? source.nFrames ?? source.posteriors.ids.length/source.posteriors.k}}),
}];
const wav = decodeWav(readFileSync(values.audio));
let audio = wav.samples;
if (wav.sampleRate !== 16000) {
  const rs = new Resampler(wav.sampleRate,16000), a = rs.push(audio), b = rs.flush();
  audio = new Float32Array(a.length+b.length); audio.set(a);audio.set(b,a.length);
}
const requests = wrongWordRechecks(chunks,index,phrases).filter(c => {
  const w = index.words[c.word]; return !values.word || `${w.surah}:${w.ayah}:${w.word}` === values.word;
});
const denoiser = new NonlocalAcousticDenoiser(audio,calibration.selected), result = [];
try {
  for (const c of requests) {
    const frameMs = chunks[0].frameMs, a = Math.round(c.fromFrame*frameMs*16), b = Math.round(c.toFrame*frameMs*16);
    if (a < 0 || b > audio.length || b <= a) continue;
    const repair = denoiser.repair(a,b);
    const raw = await replayAcousticSlot(model,audio,c,frameMs);
    const recovered = repair.patches.length ? await replayAcousticSlot(model,repair.samples,c,frameMs) : raw;
    const rawWritten = completePronunciationMatch(c.written,raw.phones), written = completePronunciationMatch(c.written,recovered.phones);
    const rival = completePronunciationMatch(c.observed,recovered.phones) || !Number.isFinite(recovered.writtenNll) ||
      !Number.isFinite(recovered.rivalNll) || recovered.writtenNll > recovered.rivalNll;
    result.push({...c,word:index.words[c.word],raw,recovered,rawWritten,written,rival,
      eligible:repair.patches.length>0,reconstructionSpecific:!rawWritten && written && !rival,patches:repair.patches});
    console.log(JSON.stringify({word:c.word,fromFrame:c.fromFrame,toFrame:c.toFrame,raw,recovered,
      patches:repair.patches.length,reconstructionSpecific:!rawWritten && written && !rival}));
  }
  writeFileSync(values.out,JSON.stringify({geometry:denoiser.geometry,requests:result},null,2));
} finally { await model.backend.release(); }
