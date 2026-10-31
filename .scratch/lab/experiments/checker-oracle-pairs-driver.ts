// Audit driver: copy to packages/core/src/bench and reconstruct checker-extra-baseline.ts from main da14389 before running on Modal.
import { readFileSync, writeFileSync } from 'node:fs';
import { createChecker } from '../checker.js';
import { createChecker as baseline } from '../checker-extra-baseline.js';
import { loadReferenceIndex } from '../reference.js';
import { chunks, oracleEvidence, oraclePositions } from './lab.js';
import { scoreSessionEvents } from './sessions.js';
const index=loadReferenceIndex(JSON.parse(readFileSync('assets/ref-v2.json','utf8')));
const phrases=JSON.parse(readFileSync('assets/phrases-v1.json','utf8'));
const manifest=JSON.parse(readFileSync('/lab/data/lab-v1.5/manifest-v1.json','utf8'));
const rows=manifest.sessions.filter((s:any)=>s.split==='practice').filter((_:any,i:number)=>i%Number(process.env.LAB_SHARDS)===Number(process.env.LAB_SHARD)).map((entry:any)=>{
 const s=entry.session, ev=oracleEvidence(s,index);
 const run=(make:typeof createChecker)=>{
  const checker=make(index,{phrases}), pos=oraclePositions(s,index);const events:any[]=[];
  for(const e of chunks(ev))events.push(...checker.push(e,pos((e.decodedTo??0)*e.frameMs)));
  events.push(...checker.end(pos(Infinity)));
  const score=scoreSessionEvents(s,events,index,{kind:'none'},{anyClass:true});
  return score.assignments.filter(a=>a.confirmed && a.eventIndex===null).map(a=>({assignment:a,events:events.filter(e=>e.type==='checker'&&e.id===a.warningId)}));
 };
 return {id:s.id,baseline:run(baseline),candidate:run(createChecker)};
});
writeFileSync(`${process.env.LAB_OUT}/oracle-pairs.json`,JSON.stringify(rows));
