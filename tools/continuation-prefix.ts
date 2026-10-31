import { readFileSync, writeFileSync } from "node:fs";
import { chunks, oraclePositions } from "../packages/core/src/bench/lab.js";
import { loadReferenceIndex } from "../packages/core/src/reference.js";
import { createTracker } from "../packages/core/src/tracker.js";
import { createChecker } from "../packages/core/src/checker.js";
function runPrefix(s:any,evidence:any,index:any,phrases:any,truth=false) {
 const tracker=createTracker(index,{mode:"free",phrases});
 const checker=createChecker(index,{phrases}); const oracle=oraclePositions(s,index); const out:any[]=[];
 for(const e of chunks(evidence)) {
  if((e.decodedTo ?? 0)*e.frameMs/1000 > Number(process.env.TRACE_CUTOFF)) break;
  const p=truth?oracle((e.decodedTo ?? 0)*e.frameMs):tracker.push({...e,posteriors:undefined});
  out.push(...p,...checker.push(e,p));
 }
 return out;
}
const rows=JSON.parse(readFileSync("/app/tools/continuation-sites.json","utf8"));
const row=rows[Number(process.env.TRACE_INDEX ?? process.env.LAB_SHARD ?? 0)];
const root="/lab/data/lab-v1.5/sessions/";
const s=JSON.parse(readFileSync(root+row.session+".json","utf8"));
const index=loadReferenceIndex(JSON.parse(readFileSync("/app/packages/core/assets/ref-v2.json","utf8")));
const phrases=JSON.parse(readFileSync("/app/packages/core/assets/phrases-v1.json","utf8"));
const models={main:"spliced-clean-int8",uniform:"phone-channel-int8",targeted:"targeted-consonants-int8",selected:"sparse-consonants-v2-int8",random:"sparse-random-v2-int8"};
const results:any={};
for(const [name,model] of Object.entries(models)) {
 const ev=JSON.parse(readFileSync(root+row.session+"."+model+".phone.evidence.json","utf8"));
 const events=runPrefix(s,ev,index,phrases);
 const near=(e:any)=>Math.abs(e.timeMs/1000-row.timeS)<12;
 const before=events.filter((e:any)=>e.type==="position" && e.timeMs/1000<=row.timeS-12).slice(-1);
 results[name]={model,engine:[...before,...events.filter(near)],truePosition:runPrefix(s,ev,index,phrases,true).filter(near)};
}
writeFileSync(process.env.LAB_OUT+"/engine.json",JSON.stringify({row,results}));
