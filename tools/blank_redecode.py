"""One preregistered acoustic-only candidate: rescue near-tie consonants from blank.
No text or session labels are used in decoding. Practice restriction is evaluation orchestration.
"""
import json,os,hashlib
from pathlib import Path
D=Path('/lab/data/lab-v1.5'); OUT=Path(os.environ['LAB_OUT'])
manifest=json.loads((D/'manifest-v1.json').read_text())
sessions=[s for s in manifest['sessions'] if s['split']=='practice']
s=sessions[int(os.environ.get('LAB_SHARD',0))]
tokens=json.loads(Path('/app/packages/core/assets/ref-v2.json').read_text())['inventory']['tokens']
consonants=set('ءبتثجحخدذرزسشصضطظعغفقكلمنهوي')
results=[]
for variant in ['phone','studio']:
 path=D/'sessions'/f"{s['id']}.spliced-clean-int8.{variant}.evidence.json"
 original=path.read_bytes();ev=json.loads(original);post=ev['posteriors'];out=[];prev=0;changed=[]
 for f in range(ev['nFrames']):
  row=list(zip(post['ids'][f*post['k']:(f+1)*post['k']],post['probs'][f*post['k']:(f+1)*post['k']]))
  tok,p=row[0]
  if tok==0:
   choices=[(i,q) for i,q in row[1:] if i and tokens[i-1] in consonants and q>=.30 and q>=.75*p]
   if choices:
    tok,p=max(choices,key=lambda iq:iq[1]);changed.append({'frame':f,'token':tokens[tok-1],'p':p,'blank':row[0][1]})
  if tok and tok==prev:
   out[-1]['end']=f;out[-1]['conf']=max(out[-1]['conf'],p)
  elif tok:out.append({'p':tokens[tok-1],'start':f,'end':f,'conf':p})
  prev=tok
 target=D/'sessions'/f"{s['id']}.blank-margin-int8.{variant}.evidence.json"
 target.write_text(json.dumps({**ev,'phonemes':out},ensure_ascii=False,separators=(',',':')))
 results.append({'session':s['id'],'variant':variant,'inputSha256':hashlib.sha256(original).hexdigest(),'changed':changed,'before':len(ev['phonemes']),'after':len(out)})
OUT.mkdir(exist_ok=True,parents=True);(OUT/'changes.json').write_text(json.dumps(results))
print(s['id'],[(x['variant'],len(x['changed'])) for x in results])
