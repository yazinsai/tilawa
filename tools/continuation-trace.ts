import { readFileSync, writeFileSync } from "node:fs";
import { runEngine, runTruePosition } from "../packages/core/src/bench/lab.js";
import { loadReferenceIndex } from "../packages/core/src/reference.js";
const rows=JSON.parse(readFileSync("/app/tools/continuation-sites.json","utf8"));
const row=rows[Number(process.env.LAB_SHARD ?? 0)];
const root="/lab/data/lab-v1.5/sessions/";
const s=JSON.parse(readFileSync(root+row.session+".json","utf8"));
const index=loadReferenceIndex(JSON.parse(readFileSync("/app/packages/core/assets/ref-v2.json","utf8")));
const phrases=JSON.parse(readFileSync("/app/packages/core/assets/phrases-v1.json","utf8"));
const models={main:"spliced-clean-int8",uniform:"phone-channel-int8",targeted:"targeted-consonants-int8",selected:"sparse-consonants-v2-int8",random:"sparse-random-v2-int8"};
const results:any={};
for(const [name,model] of Object.entries(models)) {
 const ev=JSON.parse(readFileSync(root+row.session+"."+model+".phone.evidence.json","utf8"));
 const events=runEngine(s,ev,index,phrases);
 const near=(e:any)=>Math.abs(e.timeMs/1000-row.timeS)<12;
 const before=events.filter((e:any)=>e.type==="position" && e.timeMs/1000<=row.timeS-12).slice(-1);
 results[name]={model,engine:[...before,...events.filter(near)],truePosition:runTruePosition(s,ev,index,phrases).filter(near)};
}
writeFileSync(process.env.LAB_OUT+"/engine.json",JSON.stringify({row,results}));
