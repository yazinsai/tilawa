import os,json,collections
from pathlib import Path
import lab_phonemes_core as P
out=Path(os.environ['LAB_OUT']); data=json.loads((out/'engine.json').read_text()); row=data['row']
root=Path('/lab/data/lab-v1.5/sessions'); s=json.loads((root/(row['session']+'.json')).read_text())
a,b=row['timeS']-6,row['timeS']+6
cols=P.window_columns(s,None,a,b); spans=[(c['startS'],c['endS']) for c in cols]
tokens=json.loads(Path('/app/packages/core/assets/ref-v2.json').read_text())['inventory']['tokens']
for name,run in data['results'].items():
 ev=json.loads((root/(row['session']+'.'+run['model']+'.phone.evidence.json')).read_text())
 heard=P.heard_in(ev,a,b); ops=P.align([(p,i) for i,c in enumerate(cols) for p in c['expected']],heard,spans)
 run['columns']=[]
 for i,c in enumerate(cols):
  co=[o for o in ops if o.get('col')==i]; peaks={}
  for p in set(c['expected']):
   tid=tokens.index(p)+1
   found=[]
   for frame in range(max(0,int(c['startS']*1000/ev['frameMs'])),min(ev['nFrames'],int(c['endS']*1000/ev['frameMs'])+1)):
    k=ev['posteriors']['k']; ids=ev['posteriors']['ids'][frame*k:(frame+1)*k]; probs=ev['posteriors']['probs'][frame*k:(frame+1)*k]
    if tid in ids:found.append((probs[ids.index(tid)],frame*ev['frameMs']/1000))
   peaks[p]=max(found,default=(0,None))
  run['columns'].append({**c,'ops':co,'peaks':peaks})
 run['heard']=heard
(out/'trace.json').write_text(json.dumps(data,ensure_ascii=False,indent=1))
print(json.dumps({'session':row['session'],'word':row['word'],'components':{k:v['component'] for k,v in row['arms'].items()}},ensure_ascii=False))
