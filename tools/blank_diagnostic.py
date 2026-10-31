import json, os, collections
from pathlib import Path
import lab_phonemes_core as P
D=Path('/lab/data/lab-v1.5'); OUT=Path(os.environ['LAB_OUT'])
inc=json.loads(Path('/app/tools/blank-incidents.json').read_text())['incidents']
ref=P.Ref(json.loads(Path('/app/packages/core/assets/ref-v2.json').read_text()))
tokens=json.loads(Path('/app/packages/core/assets/ref-v2.json').read_text())['inventory']['tokens']
vowels=set('َُِاويٰ')
records=[]
for x in inc:
 s=json.loads((D/'sessions'/f"{x['session']}.json").read_text())
 cols=P.window_columns(s,ref,x['timeS']-1,x['timeS']+1)
 focus=[c for c in cols if c['at']==x['word']]
 if focus: cols=focus
 a=min(c['startS'] for c in cols); b=max(c['endS'] for c in cols)
 row={'incident':x,'startS':a,'endS':b,'columns':cols}
 for variant in ['phone','studio']:
  ev=json.loads((D/'sessions'/f"{x['session']}.spliced-clean-int8.{variant}.evidence.json").read_text())
  heard=P.heard_in(ev,a-.12,b+.12)
  expected=[(c,i) for i,w in enumerate(cols) for c in w['expected']]
  ops=P.align(expected,heard,[(c['startS'],c['endS']) for c in cols])
  lo=max(0,int(a*1000/ev['frameMs'])); hi=min(ev['nFrames'],int(b*1000/ev['frameMs'])+1)
  post=ev['posteriors']; frames=[]
  for f in range(lo,hi):
   pairs=dict(zip(post['ids'][f*post['k']:(f+1)*post['k']],post['probs'][f*post['k']:(f+1)*post['k']]))
   top=post['ids'][f*post['k']]; frames.append({'frame':f,'top':tokens[top-1] if top else '_','p':pairs.get(top,0),'blank':pairs.get(0,0)})
  consonant_ops=[o for o in ops if o['op']!='same' and (o.get('e') or o.get('h')) not in vowels]
  phones=[p for p in ev['phonemes'] if a-.12<=p['start']*ev['frameMs']/1000<=b+.12]
  row[variant]={'ops':ops,'consonantOps':consonant_ops,'phones':phones,'frames':frames}
 records.append(row)
OUT.mkdir(exist_ok=True,parents=True); (OUT/'diagnostic.json').write_text(json.dumps(records,ensure_ascii=False,indent=1))
print(json.dumps({'incidents':len(records),'ops':{v:dict(collections.Counter(o['op'] for r in records for o in r[v]['consonantOps'])) for v in ['phone','studio']}}))
