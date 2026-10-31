/** Isolated human-confirmed controls; unrelated warnings are diagnostic, not truth. */
import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { loadReferenceIndex } from '@tilawa/core';
import { loadNodeModel } from '../node.js';
import { runHelpBench } from './help.js';
const {values} = parseArgs({options:{model:{type:'string'},out:{type:'string'},help:{type:'string'}}});
if (!values.model || !values.out || !values.help) throw Error('model, out, help required');
const asset=(p:string)=>JSON.parse(readFileSync(new URL(`../../../core/assets/${p}`,import.meta.url),'utf8'));
const result=await runHelpBench({helpDir:values.help,reference:loadReferenceIndex(asset('ref-v2.json')),phrases:asset('phrases-v1.json'),model:await loadNodeModel(values.model),pushMs:20,evidenceMs:3600000});
const clips=result.clips.map(({posteriors,...c})=>c);
writeFileSync(values.out,JSON.stringify({scope:'Only target misses/hits are grounded. Other warnings may concern unresolved regions; not counted as false alarms.',selfCorrections:'No verified self-correction controls available',model:values.model,...result,clips},null,1));
console.log(JSON.stringify(clips.map(c=>({id:c.id,free:c.free,declared:c.declared}))));
