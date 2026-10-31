import json,os,bisect,collections
from pathlib import Path
import lab_phonemes_core as P
D=Path('/lab/data/lab-v1.5');OUT=Path(os.environ['LAB_OUT']);manifest=json.loads((D/'manifest-v1.json').read_text());s=[s for s in manifest['sessions'] if s['split']=='practice'][int(os.environ.get('LAB_SHARD',0))]
cs=set('ءبتثجحخدذرزسشصضطظعغفقكلمنهوي');results={}
for v in ['phone','studio']:
 doc=json.loads((D/'sessions'/f"{s['id']}.json").read_text()); ev=[json.loads((D/'sessions'/f"{s['id']}.{l}.{v}.evidence.json").read_text()) for l in ['spliced-clean-int8','blank-margin-int8']]
 counts=collections.Counter();rows=[]
 for w in doc['said']:
  exp=w.get('phonemes','');a=w['startS'];b=w['endS'];types=[]
  if any(x==y and x in cs for x,y in zip(exp,exp[1:])):types.append('gemination')
  if w.get('role') in ['wrong','added','repeat','restart']:types.append(w['role'])
  errs=[]
  for e in ev:
   heard=P.heard_in(e,a-.04,b+.04);ops=P.align([(p,0) for p in exp],heard,[(a,b)],tol=.04)
   errs.append(sum(o['op']!='same' and (o.get('e') in cs or o.get('h') in cs) for o in ops))
  if types:
   rows.append({'word':P.key(w.get('at')),'startS':a,'role':w.get('role'),'types':types,'errors':errs})
   for t in types:
    counts[t+'_words']+=1;counts[t+'_errors_before']+=errs[0];counts[t+'_errors_after']+=errs[1];counts[t+'_improved']+=errs[1]<errs[0];counts[t+'_regressed']+=errs[1]>errs[0]
 for p in ev[0]['phonemes']:
  if p['p'] in cs and p['start']==p['end']:
   t=p['start']*ev[0]['frameMs']/1000
   w=next((w for w in doc['said'] if w['startS']<=t<=w['endS'] and p['p'] in w.get('phonemes','')),None)
   if w:
    counts['one_frame_as_said_consonants']+=1
    counts['one_frame_preserved']+=any(q['p']==p['p'] and q['start']<=p['start']<=q['end'] for q in ev[1]['phonemes'])
 starts=[w['startS'] for w in doc['said']];ends=[w['endS'] for w in doc['said']]
 for e,name in zip(ev,['before','after']):
  for p in e['phonemes']:
   t=p['start']*e['frameMs']/1000;i=bisect.bisect_right(starts,t+.12)-1
   if i<0 or ends[i]<t-.12:counts['pause_phones_'+name]+=1
 results[v]={'counts':dict(counts),'wordControls':rows}
OUT.mkdir(exist_ok=True,parents=True);(OUT/'controls.json').write_text(json.dumps({'session':s['id'],'variants':results}))
print(s['id'],{v:r['counts'] for v,r in results.items()})
