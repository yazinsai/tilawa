"""Scout a frozen sparse candidate pool; does not train or change labels."""
import json
from pathlib import Path
import modal
import student_train as S
ROOT=Path(__file__).resolve().parent.parent
image=(S.image.add_local_python_source('student_train')
       .add_local_file(ROOT/'tools/sparse_mining_core.py','/root/tools/sparse_mining_core.py')
       .add_local_file(ROOT/'tools/sparse-word-metadata.json.gz','/root/sparse-word-metadata.json.gz'))
app=modal.App('tilawa-sparse-mine',image=image)
@app.function(gpu='H100',cpu=12,memory=131072,volumes={'/student':S.vol,'/cache':S.cache_vol},timeout=7200)
def mine():
 import gzip,hashlib,io,random
 import numpy as np
 import soundfile as sf
 import torch
 import sparse_mining_core as M
 from student import phoneme_ids,TOKENS
 torch.manual_seed(0)
 old=torch.load('/student/runs/paired-control-v1/ckpt.pt',map_location='cpu',weights_only=False)
 cfg={**S.DEFAULTS,**old['cfg'],'targeted':False}; init=cfg['init_run'];del old
 ck=torch.load(f'/student/runs/{init}/ckpt.pt',map_location='cpu',weights_only=False)
 model=S.build_model('full').cuda(); model.load_state_dict(ck['model']);del ck
 model.eval();model.streaming_eval(16,128)
 c=S.load_clips('clean-match','train',50.,30.,0,exclude_gold=True)
 metadata=json.load(gzip.open('/root/sparse-word-metadata.json.gz','rt'))
 aug={k:dict(np.load(f'/student/aug/{k}.npz')) for k in ('noise','rir')}
 scout=sorted(random.Random(2718).sample(range(len(c['key'])),min(2048,len(c['key']))))
 print(json.dumps({'event':'loaded','pool':len(c['key']),'scout':len(scout),'init':init}),flush=True)
 def decode(phone,seed):
  ds=S.ClipDataset(c,{**cfg,'phone_p':phone,'seed':seed,'spec_aug':False},train=bool(phone),aug=aug)
  out={}
  batches=[];batch=[];seconds=0
  for i in sorted(scout,key=lambda i:c['n_samples'][i]):
   t=c['n_samples'][i]/16000
   if batch and seconds+t>200:batches.append(batch);batch=[];seconds=0
   batch.append(i);seconds+=t
  if batch:batches.append(batch)
  loader=torch.utils.data.DataLoader(ds,batch_sampler=batches,collate_fn=S.collate,num_workers=8)
  with torch.no_grad():
   for b in loader:
    with torch.autocast('cuda',dtype=torch.bfloat16):lp,lens=model(b['feats'].cuda(),b['feat_lens'].cuda())
    best=lp.argmax(-1).cpu()
    for j,i in enumerate(b['i'].tolist()):out[i]=M.runs(best[j,:lens[j]].tolist())
  print(json.dumps({'event':'decoded','phone':phone,'seed':seed}),flush=True)
  return out
 run=Path('/student/runs/sparse-mining-v1');run.mkdir(exist_ok=True)
 cache=run/'decode-cache.json'
 if cache.exists():
  saved=json.loads(cache.read_text());clean={int(k):v for k,v in saved['clean'].items()};degraded=[{int(k):v for k,v in d.items()} for d in saved['degraded']]
 else:
  clean=decode(0.,0);degraded=[decode(1.,seed) for seed in (101,202,303)]
  cache.write_text(json.dumps({'clean':clean,'degraded':degraded}));S.vol.commit()
 rows=[];eligible=[];candidates=[]
 for i in scout:
  key=c['key'][i];meta=metadata.get(key)
  if not meta or meta['sourcePer']!=0 or meta['sourceConfidence']<.99:continue
  target=M.word_target(meta['words'],phoneme_ids);timed=M.attach_timing(target,clean[i])
  if timed is None:continue
  lo,hi=c['top_off'][i:i+2]
  teacher_ids=c['ids'][lo:hi].reshape(-1,5)[:,0].tolist()
  if [r['token'] for r in M.duration_groups(M.runs(teacher_ids))]!=[r['token'] for r in target]:continue
  views=[M.failures(timed,d[i]) for d in degraded]
  bad=M.persistent(timed,views,set(range(1,29)))
  # Preserve source word text and occurrence identity for independent review.
  row={'key':key,'index':i,'words':meta['words'],'timed':timed,'views':views,'persistent':bad,
       'repeatedWords':[j for j in range(1,len(meta['words'])) if meta['words'][j]['text']==meta['words'][j-1]['text']],
       'trusted':False,'verification':'pending independent audio review'}
  rows.append(row);eligible.append(key)
  if bad:candidates.append(key)
 print(json.dumps({'event':'eligibility','eligible':len(eligible),'candidates':len(candidates)}),flush=True)
 (run/'eligibility.json').write_text(json.dumps({'rows':rows,'eligible':eligible,'candidates':candidates}));S.vol.commit()
 arms=M.freeze_arms(eligible,candidates,cap=128,fraction=1.,seed=2718)
 run=Path('/student/runs/sparse-mining-v1');run.mkdir(exist_ok=True)
 chosen=set(arms['selected']+arms['random'])
 for r in rows:
  if r['key'] not in chosen:continue
  i=r['index'];lo,hi=c['flac_off'][i:i+2];flac=c['flac'][lo:hi].tobytes()
  r['audioSha256']=hashlib.sha256(flac).hexdigest()
  (run/(r['key'].replace(':','_')+'.flac')).write_bytes(flac)
 report={'policy':{'scout':2048,'seeds':[101,202,303],'persistent':'same occurrence fails in >=2/3 views','cap':128,'weight':3,'updates':200,'ctcWeight':1,'kdWeight':2,'trainingStarted':False},'initRun':init,'poolKeys':c['key'],'scoutKeys':[c['key'][i] for i in scout],'arms':arms,'rows':rows}
 (run/'mining.json').write_text(json.dumps(report,ensure_ascii=False,indent=1));S.vol.commit()
 return {'pool':len(c['key']),'eligible':len(eligible),'persistent':len(candidates),'selected':arms['n'],'path':str(run/'mining.json')}
@app.local_entrypoint()
def main():
 print(json.dumps(mine.remote(),indent=1))

@app.function(cpu=2,memory=8192,volumes={'/student':S.vol,'/lab':modal.Volume.from_name('tilawa-lab')},timeout=600)
def prepare_review():
 import io,soundfile as sf
 run=Path('/student/runs/sparse-mining-v1')
 report=json.loads((run/'mining.json').read_text())
 root=Path('/lab/data/sparse-training-review');root.mkdir(exist_ok=True)
 chosen=set(report['arms']['selected']+report['arms']['random']);items=[]
 for r in report['rows']:
  if r['key'] not in chosen:continue
  name=r['key'].replace(':','_');wave,sr=sf.read(run/(name+'.flac'),dtype='float32')
  sf.write(root/(name+'.wav'),wave,sr,subtype='PCM_16')
  items.append({'id':r['key'],'audio':{'path':'sparse-training-review/'+name+'.wav','startS':0.,'endS':len(wave)/sr}})
 (root/'items.json').write_text(json.dumps(items,indent=1));(root/'mining.json').write_text(json.dumps(report,ensure_ascii=False))
 modal.Volume.from_name('tilawa-lab').commit()
 return items

@app.local_entrypoint()
def review_inputs():
 items=prepare_review.remote()
 Path('.scratch/lab/experiments/sparse-training-review-items.json').write_text(json.dumps(items,indent=1))
