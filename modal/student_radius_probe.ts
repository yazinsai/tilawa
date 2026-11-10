/** Generic acoustic premise probe. No labels, target locations or scoring decisions are supplied. */
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "@tilawa/core";
import type { PhonemeEvidence } from "@tilawa/core";
import { loadNodeModel, decodeWav, Resampler } from "../packages/sdk/src/node.js";
import { estimateRadius, radiusCandidates } from "../packages/sdk/src/acoustic-radius.js";
import type { RadiusCalibration } from "../packages/sdk/src/acoustic-radius.js";

const {values}=parseArgs({options:{model:{type:"string"},audio:{type:"string"},evidence:{type:"string"},
  calibration:{type:"string"},out:{type:"string"}}});
if (!values.model || !values.audio || !values.evidence || !values.calibration || !values.out)
  throw new Error("--model, --audio, --evidence, --calibration and --out required");
const model=await loadNodeModel(values.model);
const calibration=JSON.parse(readFileSync(values.calibration,"utf8")) as RadiusCalibration;
if (calibration.schema!==1 || calibration.primarySha256!==model.manifest.sha256 || calibration.inventory!==model.manifest.inventory)
  throw new Error("Probe calibration differs from primary listener");
const asset=(file:string)=>JSON.parse(readFileSync(new URL(`../packages/core/assets/${file}`,import.meta.url),"utf8"));
const index=loadReferenceIndex(asset("ref-v2.json")), phrases=asset("phrases-v1.json");
const stored=JSON.parse(readFileSync(values.evidence,"utf8"));
const chunks:PhonemeEvidence[]=Array.isArray(stored)?stored:[{...stored,inventory:stored.inventory??index.inventoryId,
  posteriors:stored.posteriors?{...stored.posteriors,startFrame:stored.posteriors.startFrame??0,nFrames:stored.nFrames??stored.posteriors.nFrames}:undefined}];
const wav=decodeWav(readFileSync(values.audio));
let audio=wav.samples;
if (wav.sampleRate!==16000) {
  const rs=new Resampler(wav.sampleRate,16000), body=rs.push(audio), tail=rs.flush();
  audio=new Float32Array(body.length+tail.length);audio.set(body);audio.set(tail,body.length);
}
const results=[];
for (const c of radiusCandidates(chunks,index,phrases)) {
  const result=await estimateRadius(model,audio,chunks,c,calibration.search);
  results.push({candidate:c,result,veto:result.energy!==null&&result.energy<calibration.rejectionEnergy});
}
writeFileSync(values.out,JSON.stringify({schema:1,primarySha256:model.manifest.sha256,calibration,results},null,2));
await model.backend.release();
