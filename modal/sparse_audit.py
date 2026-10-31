"""Frozen-input objective audit, no updates or validation-based choices."""
import json
from pathlib import Path
import modal
import student_train as S
import paired_source as P
app=modal.App('tilawa-sparse-audit',image=S.image.add_local_python_source('student_train','paired_source'))
@app.function(gpu='H100',cpu=8,memory=65536,volumes={'/student':S.vol,'/cache':S.cache_vol},timeout=1800)
def audit():
 import numpy as np
 import torch
 from student import spliced_target
 torch.manual_seed(0)
 cfg={**P.DEFAULTS,'size':'full','target':'spliced','kd_weight':2.,'ctc_weight':1.,'pair_weight':.5,'phone_p':.75,'seed':0}
 # Exact original initialization, discovered from saved training config.
 old=torch.load('/student/runs/paired-control-v1/ckpt.pt',map_location='cpu',weights_only=False)
 init=old['cfg']['init_run']; del old
 ck=torch.load(f'/student/runs/{init}/ckpt.pt',map_location='cpu',weights_only=False)
 model=P.build_model('full').cuda(); model.load_state_dict(ck['model']); del ck
 model.eval(); model.streaming_eval(16,128)
 clips=S.load_clips('clean-match','train',0,30.,0,12,exclude_gold=True)
 clips['target']=[spliced_target(r,h) for r,h in zip(clips['ref'],clips['hyp'])]
 aug={k:dict(np.load(f'/student/aug/{k}.npz')) for k in ('noise','rir')}
 ds=P.ClipDataset(clips,cfg,True,aug)
 params=[p for p in model.parameters() if p.requires_grad]
 def grad(x):
  return [g.detach() if g is not None else None for g in torch.autograd.grad(x,params,retain_graph=True,allow_unused=True)]
 def dot(a,b): return sum((x.double()*y.double()).sum().item() for x,y in zip(a,b) if x is not None and y is not None)
 def stats(a,b):
  na=dot(a,a)**.5; nb=dot(b,b)**.5
  return {'normA':na,'normB':nb,'ratio':na/max(nb,1e-30),'cosine':dot(a,b)/max(na*nb,1e-30)}
 rows=[]
 for ix in ([0,1],[4,5],[8,9]):
  b=P.collate([ds[i] for i in ix]); out=P.losses(model,b,cfg,'cuda')
  gt=grad(out['ctc']); gk=grad(2*out['kd']); gc=grad(out['clean_ctc']); gp=grad(.5*out['paired_kl'])
  r={'keys':[clips['key'][i] for i in ix],'losses':{k:float(v.detach()) for k,v in out.items()},'ctcVsWeightedKD':stats(gt,gk),'cleanCtcVsWeightedKD':stats(gc,gk),'weightedKLvsWeightedKD':stats(gp,gk),'ctcVsCleanCtc':stats(gt,gc)}
  del out,gt,gk,gc,gp
  b['clean_feats']=b['feats']; ident=P.losses(model,b,cfg,'cuda')
  baseline=ident['ctc']+2*ident['kd']; legacy=ident['ctc']+ident['clean_ctc']+2*ident['kd']; corrected=.5*(ident['ctc']+ident['clean_ctc'])+2*ident['kd']
  gb=grad(baseline); gl=grad(legacy); gf=grad(corrected)
  diff=[None if a is None or z is None else a-z for a,z in zip(gf,gb)]
  r['identical']={'kl':float(ident['paired_kl'].detach()),'ctcDifference':float((ident['ctc']-ident['clean_ctc']).detach()),'legacyVsBase':stats(gl,gb),'correctedVsBase':stats(gf,gb),'correctedRelativeGradientError':dot(diff,diff)**.5/max(dot(gb,gb)**.5,1e-30)}
  rows.append(r); print(json.dumps(r),flush=True)
  Path('/student/runs/sparse-objective-audit').mkdir(exist_ok=True)
  Path('/student/runs/sparse-objective-audit/partial.json').write_text(json.dumps(rows,indent=1)); S.vol.commit()
  del ident,gb,gl,gf,diff
 # Inventory only: never treat unreviewed takes as verified controls.
 result={'initRun':init,'cfg':cfg,'batches':rows,'trainingUpdates':0}
 Path('/student/runs/sparse-objective-audit').mkdir(exist_ok=True)
 Path('/student/runs/sparse-objective-audit/result.json').write_text(json.dumps(result,indent=1)); S.vol.commit()
 return result
@app.local_entrypoint()
def main():
 result=audit.remote(); Path('.scratch/lab/experiments/sparse-objective-audit.json').write_text(json.dumps(result,indent=1))

@app.function(gpu='H100',cpu=4,memory=32768,volumes={'/student':S.vol},timeout=900)
def gradient_identity(amp: bool = False, disable_aux: bool = False):
 import inspect,random
 import numpy as np
 import pyarrow.parquet as pq
 import torch
 import scaling
 from student import phoneme_ids,spliced_target
 source=inspect.getsource(scaling)
 interesting=[(i+1,l.strip()) for i,l in enumerate(source.splitlines()) if ('random' in l or 'randn' in l) and not l.strip().startswith('#')]
 old=torch.load('/student/runs/paired-control-v1/ckpt.pt',map_location='cpu',weights_only=False)
 init=old['cfg']['init_run'];del old
 ck=torch.load(f'/student/runs/{init}/ckpt.pt',map_location='cpu',weights_only=False)
 model=P.build_model('full').cuda();model.load_state_dict(ck['model']);del ck
 model.eval();model.streaming_eval(16,128)
 disabled=[]
 if disable_aux:
  for name,m in model.named_modules():
   if isinstance(m,(scaling.Balancer,scaling.Whiten)):
    m.forward=lambda x:x;disabled.append(name)
  # WithLoss injects an unscaled auxiliary gradient once per forward graph.
  scaling.WithLoss.backward=staticmethod(lambda ctx,g:(g,torch.zeros(ctx.y_shape,dtype=g.dtype,device=g.device),None))
  disabled.append('WithLoss.backward auxiliary branch')
  # Diagnostic-only exact derivatives for the stochastic quantized k2 activations.
  for name,m in model.named_modules():
   if isinstance(m,scaling.SwooshL):
    m.forward=lambda x:torch.nn.functional.softplus(x-4.)-.08*x-.035
    disabled.append(name+':native-SwooshL')
   elif isinstance(m,scaling.SwooshR):
    m.forward=lambda x:torch.nn.functional.softplus(x-1.)-.08*x-.313261687
    disabled.append(name+':native-SwooshR')
   elif isinstance(m,scaling.DoubleSwish):
    m.forward=lambda x:x*torch.sigmoid(x-1.)
    disabled.append(name+':native-DoubleSwish')
 # Reconstruct the first previously audited batch directly, without rescanning all shards.
 import io,soundfile as sf,torchaudio
 items=[]
 for shard,row in [('clean-00006-of-00411',932),('clean-00011-of-00411',908)]:
  r=pq.read_table('/student/s1/'+shard+'.parquet',filters=[('row','=',row)]).to_pylist()[0]
  w,_=sf.read(io.BytesIO(r['flac']),dtype='float32')
  feat=torchaudio.compliance.kaldi.fbank(torch.from_numpy(w).unsqueeze(0),num_mel_bins=80,sample_frequency=16000,dither=0.,energy_floor=1e-10,high_freq=-400)
  items.append(dict(i=len(items),feats=feat,clean_feats=feat,target=torch.tensor(phoneme_ids(spliced_target(r['ref'],r['hyp']))),top_ids=torch.from_numpy(np.frombuffer(r['top5_ids'],np.uint8).reshape(-1,5).astype(np.int64)),top_p=torch.from_numpy(np.frombuffer(r['top5_p'],np.float16).reshape(-1,5).astype(np.float32))))
 b=P.collate(items);cfg={**P.DEFAULTS,'kd_weight':2.,'ctc_weight':1.,'pair_weight':0.,'audit_amp':amp,'audit_replay_rng':True}
 out=P.losses(model,b,cfg,'cuda');params=list(model.parameters())
 def grad(x,seed=None):
  if seed is not None:random.seed(seed);torch.manual_seed(seed)
  return [g.detach() if g is not None else None for g in torch.autograd.grad(x,params,retain_graph=True,allow_unused=True)]
 def err(a,b):
  def norm(x):return sum((y.double().square().sum().item() if y is not None else 0) for y in x)**.5
  diff=[None if x is None else x-y for x,y in zip(a,b)]
  return {'all':norm(diff)/norm(b),'head':norm(diff[-2:])/norm(b[-2:])}
 base=out['ctc']+2*out['kd'];averaged=.5*(out['ctc']+out['clean_ctc'])+2*out['kd']
 a=grad(base);b1=grad(base);unseed=err(a,b1);del a,b1
 a=grad(base,123);b1=grad(base,123);seeded=err(a,b1);del a,b1
 a=grad(base,123);b1=grad(averaged,123);paired=err(b1,a);del a,b1
 def fresh(which):
  model.zero_grad(set_to_none=True);random.seed(456);torch.manual_seed(456)
  q=P.losses(model,b,cfg,'cuda')
  loss=q['ctc']+2*q['kd'] if which=='base' else .5*(q['ctc']+q['clean_ctc'])+2*q['kd']
  loss.backward()
  return [p.grad.detach().clone() if p.grad is not None else None for p in params]
 fresh_a=fresh('base');fresh_b=fresh('base');fresh_repeat=err(fresh_b,fresh_a);del fresh_b
 fresh_b=fresh('average');fresh_average=err(fresh_b,fresh_a);del fresh_a,fresh_b
 result={'freshSingleViewRepeat':fresh_repeat,'freshAveragedViewsVsSingle':fresh_average,'amp':amp,'disabledAux':disabled,'sourceRandomness':interesting,'repeatBackward':unseed,'seededRepeatBackward':seeded,'seededAveragedViewsVsBase':paired}
 Path('/student/runs/sparse-objective-audit/identity.json').write_text(json.dumps(result,indent=1));S.vol.commit();return result
@app.local_entrypoint()
def identity():
 r=gradient_identity.remote();Path('.scratch/lab/experiments/sparse-gradient-fresh-graphs.json').write_text(json.dumps(r,indent=1));print(json.dumps(r,indent=1))

@app.function(cpu=1,memory=2048,timeout=120)
def scaling_source():
 import inspect,scaling
 s=inspect.getsource(scaling).splitlines()
 return '\n'.join(f'{i+1}: {line}' for i,line in enumerate(s) if i<=1660)
@app.local_entrypoint()
def inspect_scaling():
 Path('.scratch/lab/experiments/scaling-gradient-source.txt').write_text(scaling_source.remote())

@app.function(gpu='H100',cpu=4,memory=32768,volumes={'/student':S.vol},timeout=600)
def head_audit():
 import io
 import numpy as np
 import pyarrow.parquet as pq
 import soundfile as sf
 import torch
 import torchaudio
 import torch.nn.functional as F
 from student import phoneme_ids,spliced_target,soft_target_loss
 old=torch.load('/student/runs/paired-control-v1/ckpt.pt',map_location='cpu',weights_only=False)
 ck=torch.load('/student/runs/'+old['cfg']['init_run']+'/ckpt.pt',map_location='cpu',weights_only=False);del old
 model=P.build_model('full').cuda();model.load_state_dict(ck['model']);del ck
 model.eval();model.streaming_eval(16,128)
 aug={k:dict(np.load(f'/student/aug/{k}.npz')) for k in ('noise','rir')}
 pairs=[ [('clean-00006-of-00411',932),('clean-00011-of-00411',908)], [('clean-00243-of-00411',531),('clean-00256-of-00411',776)], [('clean-00381-of-00411',417),('clean-00388-of-00411',697)] ]
 head=model.ctc_output[-1];captures=[]
 hook=head.register_forward_pre_hook(lambda m,args:captures.append(args[0].detach()))
 result=[]
 for pair in pairs:
  rows=[pq.read_table('/student/s1/'+sh+'.parquet',filters=[('row','=',row)]).to_pylist()[0] for sh,row in pair]
  # Repack these fixed clips; use the same original index for deterministic phone augmentation.
  c={k:[r[k] for r in rows] for k in ('ref','hyp','n_frames','n_samples')}
  for key,field in [('flac','flac'),('ids','top5_ids'),('probs','top5_p')]:
   dt=np.uint8 if key!='probs' else np.float16
   arrays=[np.frombuffer(r[field],dt) for r in rows];c[key]=np.concatenate(arrays)
   if key!='probs':c['flac_off' if key=='flac' else 'top_off']=np.r_[0,np.cumsum([len(v) for v in arrays])]
  c['target']=[spliced_target(r['ref'],r['hyp']) for r in rows]
  cfg={**P.DEFAULTS,'phone_p':.75,'seed':0,'spec_aug':False}
  ds=P.ClipDataset(c,cfg,True,aug);b=P.collate([ds[i] for i in range(2)])
  feats=b['feats'].cuda();lens=b['feat_lens'].cuda()
  with torch.no_grad():_,ol=model(feats,lens);_,cl=model(b['clean_feats'].cuda(),lens)
  hn,hc=captures;captures.clear()
  lp=F.linear(hn,head.weight,head.bias).log_softmax(-1);cp=F.linear(hc,head.weight,head.bias).log_softmax(-1)
  targets=b['targets'].cuda();tl=b['target_lens'].cuda();frames=ol.sum()
  ctc=F.ctc_loss(lp.transpose(0,1),targets,ol,tl,blank=0,reduction='sum',zero_infinity=True)/frames
  clean_ctc=F.ctc_loss(cp.transpose(0,1),targets,cl,tl,blank=0,reduction='sum',zero_infinity=True)/frames
  B,T,V=lp.shape;idx=torch.minimum(torch.arange(T,device='cuda')[None].expand(B,T),(b['n_teacher'].cuda()-1)[:,None])
  ids=b['top_ids'].cuda().gather(1,idx[...,None].expand(B,T,5));pr=b['top_p'].cuda().gather(1,idx[...,None].expand(B,T,5))
  valid=torch.arange(T,device='cuda')[None]<ol[:,None];kd=2*soft_target_loss(lp,ids,pr,valid)
  def g(loss):return torch.cat([v.flatten() for v in torch.autograd.grad(loss,(head.weight,head.bias),retain_graph=True)])
  gt,gc,gk=g(ctc),g(clean_ctc),g(kd)
  def stats(a,b):return {'norm':float(a.norm()),'ratioToKD':float(a.norm()/b.norm()),'cosineToKD':float(F.cosine_similarity(a,b,dim=0))}
  identical=.5*(ctc+ctc)+kd;baseline=ctc+kd
  err=float((g(identical)-g(baseline)).norm()/g(baseline).norm())
  result.append({'keys':[f'{s}:{r}' for s,r in pair],'noisyCTC':stats(gt,gk),'cleanCTC':stats(gc,gk),'weightedKDNorm':float(gk.norm()),'normalizedIdenticalHeadError':err})
  assert err<1e-6
 hook.remove();return {'scope':'Projection-layer gradients, frozen encoder features, FP32 head; no stochastic encoder backward modifiers. Phone augmentation fixed within each audited batch.','batches':result}
@app.local_entrypoint()
def projection():
 r=head_audit.remote();Path('.scratch/lab/experiments/sparse-projection-audit.json').write_text(json.dumps(r,indent=1));print(json.dumps(r,indent=1))
