"""Fixed-recipe fine-tuning on real continuous same-reader multi-ayah audio.

Training accepts only clean-match/train s1 recordings containing consecutive
complete canonical ayahs. Audio is never spliced. Evaluation labels are absent
from this module. Export retains the deployed chunk/cache geometry.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from student_pause_mask import PauseStreaming,stream

SR=16000


def collapse(s):
    return ''.join(p for i,p in enumerate(s) if i==0 or p!=s[i-1])


def continuous_catalog(out,max_hours=20):
    import ahocorasick
    import pyarrow.parquet as pq
    from student import held_out,exclusions
    ref=json.loads(Path('/app/packages/core/assets/ref-v2.json').read_text())
    gold=exclusions(Path('/app/docs/benchmarks/gold-per-v1-exclude.txt').read_text())
    trie=ahocorasick.Automaton()
    ayahs=ref['ayahs']
    patterns={}
    for left,right in zip(ayahs,ayahs[1:]):
        if left['s']!=right['s'] or left['a']+1!=right['a']: continue
        pattern=collapse(left['p']+right['p'])
        patterns.setdefault(pattern,[]).append(len(collapse(left['p'])))
    for pattern,ends in patterns.items(): trie.add_word(pattern,(pattern,ends))
    trie.make_automaton()
    rows=[]
    clusters=set()
    for path in sorted(Path('/student/s1').glob('*.parquet')):
        table=pq.read_table(path,columns=['shard','row','bucket','split','ref','cluster','n_samples'])
        for offset,r in enumerate(table.to_pylist()):
            if r['bucket']!='clean-match' or r['split']!='train' or r['n_samples']>60*SR or held_out(gold,r['shard'],r['row'],r['split']): continue
            normalized=collapse(r['ref']); boundaries=set()
            for end,(pattern,ends) in trie.iter(normalized):
                start=end+1-len(pattern)
                boundaries.update(start+v for v in ends)
            if not boundaries: continue
            # Convert normalized canonical boundaries back to the original phone clock.
            indices=[i for i,p in enumerate(r['ref']) if i==0 or p!=r['ref'][i-1]]
            boundaries=[indices[b] for b in sorted(boundaries) if b<len(indices)]
            rows.append(dict(path=str(path),offset=offset,boundaries=boundaries,cluster=r['cluster'],samples=r['n_samples']))
            clusters.add(r['cluster'])
    rng=np.random.default_rng(2026); rng.shuffle(rows)
    selected=[]; samples=0
    for row in rows:
        selected.append(row); samples+=row['samples']
        if samples>=max_hours*SR*3600: break
    out.mkdir(parents=True,exist_ok=True)
    report=dict(eligible_recordings=len(rows),eligible_clusters=len(clusters),selected_recordings=len(selected),hours=samples/SR/3600,selected=selected)
    (out/'catalog.json').write_text(json.dumps(report))
    print(json.dumps({k:v for k,v in report.items() if k!='selected'}),flush=True)
    return selected


def alignment_indices(lp,phones):
    """CTC Viterbi returns canonical phone index at each frame, blank=-1."""
    labels=[0]; indexes=[-1]
    for i,p in enumerate(phones): labels.extend([int(p),0]); indexes.extend([i,-1])
    scores=np.full(len(labels),-np.inf); scores[0]=0; backs=[]
    skip=np.array([i>1 and p!=0 and p!=labels[i-2] for i,p in enumerate(labels)])
    for row in lp.detach().cpu().numpy():
        one=np.r_[-np.inf,scores[:-1]]; two=np.r_[[-np.inf,-np.inf],scores[:-2]] if len(scores)>1 else np.array([-np.inf])
        two[~skip]=-np.inf
        choices=np.stack([scores,one,two]); best=choices.argmax(0)
        scores=choices[best,np.arange(len(scores))]+row[labels]; backs.append(best)
    terminal=len(labels)-1
    if scores[-2]>scores[-1]: terminal-=1
    result=np.full(len(backs),-1)
    if np.isfinite(scores[terminal]):
        for f in range(len(backs)-1,-1,-1): result[f]=indexes[terminal]; terminal-=int(backs[f][terminal])
    return result


def pause_floors(lp,phones,boundaries,rng,probability=.5):
    """Frozen-stream labelled blank pauses at canonical ayah seams, 300 ms.

    Mask starts at pause onset, preserving its frames as encoder warm-up. The
    alignment only associates a real detector pause with its generic seam.
    """
    labels=alignment_indices(lp,phones)
    blank=(lp[:,0]>=np.log(.5)).detach().cpu().numpy()
    floors=np.full(len(lp)*2+16,-1,dtype=np.int64)
    pauses=[]; start=0; latest=-1
    while start<len(blank):
        end=start+1
        while end<len(blank) and blank[end]==blank[start]: end+=1
        if blank[start] and end-start>=8:
            before=labels[:start]; after=labels[end:]; before=before[before>=0]; after=after[after>=0]
            if len(before) and len(after) and any(before[-1]<b<=after[0] for b in boundaries):
                pauses.append((start,end))
                if rng.random()<probability:
                    latest=2*start; floors[latest:]=latest
        start=end
    return torch.from_numpy(floors).to(lp.device),pauses


class Dataset:
    def __init__(self,catalog,banks,cfg): self.rows,self.banks,self.cfg=catalog,banks,cfg; self.tables={}
    def get(self,i,step):
        import pyarrow.parquet as pq
        import soundfile as sf
        import torchaudio
        from student import mix_noise,phoneme_ids
        import student_train as T
        from student_export import FBANK
        row=self.rows[i]; path=row['path']
        if path not in self.tables: self.tables[path]=pq.read_table(path,columns=['flac','ref','n_frames','top5_ids','top5_p'])
        r=self.tables[path].slice(row['offset'],1).to_pylist()[0]
        wave,rate=sf.read(io.BytesIO(r['flac']),dtype='float32')
        if rate!=SR or wave.ndim!=1: raise ValueError('expected original mono 16 kHz audio')
        rng=np.random.default_rng(np.random.SeedSequence([2026,step,i]))
        if rng.random()<self.cfg['rir_p']: wave=T.reverb(wave,T.pick(self.banks['rir'],rng))
        if rng.random()<self.cfg['noise_p']: wave=mix_noise(wave,T.pick(self.banks['noise'],rng),rng.uniform(*self.cfg['snr_db']),rng)
        wave=np.ascontiguousarray(wave*10**(rng.uniform(-10,6)/20),dtype=np.float32)
        feats=torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave).unsqueeze(0),**FBANK)
        if self.cfg['spec_aug']: feats=T.spec_augment(feats,rng)
        return dict(feats=feats,phones=phoneme_ids(r['ref']),boundaries=row['boundaries'],
                    ids=torch.from_numpy(np.frombuffer(r['top5_ids'],np.uint8).copy().reshape(-1,5).astype(np.int64)),
                    probs=torch.from_numpy(np.frombuffer(r['top5_p'],np.float16).copy().reshape(-1,5).astype(np.float32)))


def train(args):
    import student_train as T
    from student import soft_target_loss
    torch.set_num_threads(4); torch.manual_seed(2026)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    base=torch.load('/student/prior-runs/spliced-clean/ckpt.pt',map_location='cpu',weights_only=False)
    cfg=base['cfg']; raw=T.build_model(cfg['size']); raw.load_state_dict(base['model'])
    anchor=PauseStreaming(copy.deepcopy(raw)).cuda().eval().requires_grad_(False)
    model=PauseStreaming(raw).cuda().eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5)
    current=0
    if (out/'training.pt').exists():
        ck=torch.load(out/'training.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['opt']); current=ck['step']
    catalog=json.loads((out/'catalog.json').read_text())['selected'] if (out/'catalog.json').exists() else continuous_catalog(out,args.hours)
    if not catalog: raise RuntimeError('no eligible real continuous recordings')
    banks={k:dict(np.load('/student/aug/'+k+'.npz')) for k in ('noise','rir')}
    ds=Dataset(catalog,banks,cfg); rng=np.random.default_rng(2026)
    log=open(out/'log.jsonl','a'); started=time.time()
    for step in range(current,args.steps):
        i=int(rng.integers(len(catalog)))
        b=ds.get(i,step); feats=b['feats'][None].cuda()
        with torch.no_grad(): shipped=stream(anchor,feats)[0]
        cut,pauses=pause_floors(shipped,b['phones'],b['boundaries'],np.random.default_rng([2026,step]))
        lp=stream(model,feats,cut[None])[0].float(); n=len(lp)
        phones=torch.tensor(b['phones'],dtype=torch.long,device='cuda')
        ctc=F.ctc_loss(lp[:,None],phones,torch.tensor([n]),torch.tensor([len(phones)]),reduction='sum',zero_infinity=True)/n
        teacher_idx=torch.arange(n).clamp(max=len(b['ids'])-1)
        kd=soft_target_loss(lp[None],b['ids'][teacher_idx][None].cuda(),b['probs'][teacher_idx][None].cuda(),torch.ones(1,n,dtype=torch.bool,device='cuda'))
        unmasked=(cut[:2*n:2]<0)
        kl=F.kl_div(lp,shipped.detach(),log_target=True,reduction='none').sum(-1)
        # The anchor is on unmasked items; masked items learn the new regime.
        kl=kl.mean() if bool(unmasked.all()) else kl.sum()*0
        loss=ctc+2*kd+4*kl
        if not torch.isfinite(loss): raise RuntimeError('non-finite training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if step%10==0:
            record=dict(step=step+1,ctc=float(ctc),kd=float(kd),kl=float(kl),pauses=len(pauses),masked_frames=int((~unmasked).sum()),seconds=time.time()-started)
            print(json.dumps(record),flush=True); log.write(json.dumps(record)+'\n'); log.flush()
        if (step+1)%25==0 or step+1==args.steps:
            torch.save(dict(model=model.state_dict(),opt=optimizer.state_dict(),step=step+1),out/'training.tmp'); (out/'training.tmp').replace(out/'training.pt')
        if time.time()-started>args.minutes*60: break
    plain={k:v.cpu() for k,v in model.state_dict().items()}
    torch.save(dict(model=plain,cfg=cfg,state=dict(step=step+1)),out/'ckpt.pt')


def export(args):
    import student_train as T
    from student_export import manifest
    from icefall.utils import get_onnx_export_kwargs
    from onnxruntime.quantization import quantize_dynamic,QuantType
    out=Path(args.out); ck=torch.load(out/'ckpt.pt',map_location='cpu',weights_only=False)
    raw=T.build_model(ck['cfg']['size']); raw.load_state_dict(ck['model'])
    model=PauseStreaming(raw).eval(); init=model.init_states()
    kinds=['cached_key','cached_nonlin_attn','cached_val1','cached_val2','cached_conv1','cached_conv2']
    names=[f'{kinds[i%6]}_{i//6}' for i in range(len(init)-2)]+['embed_states','processed_lens']
    specs=[dict(name=n,shape=list(v.shape),dtype='int64' if v.dtype==torch.int64 else 'float32') for n,v in zip(names,init)]
    directory=out/'masked'; directory.mkdir(exist_ok=True)
    fp=directory/'float.onnx'; path=directory/'model.onnx'
    with torch.no_grad(): torch.onnx.export(model,(torch.randn(1,45,80),init,torch.full((1,16),-1,dtype=torch.int64)),str(fp),input_names=['x',*names,'pause_floor'],output_names=['log_probs',*[f'new_{n}' for n in names]],opset_version=17,**get_onnx_export_kwargs())
    quantize_dynamic(str(fp),str(path),weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul'])
    m=manifest('pause-mask-trained',specs,16,128,str(path),ck); m['format']='onnx-int8'; m['pauseMask']=dict(input='pause_floor',clockMs=20)
    (directory/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=1))
    print(json.dumps(dict(export=str(path),sha256=m['sha256'],bytes=m['bytes'],step=ck['state']['step'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['catalog','train','export']); p.add_argument('--out',required=True); p.add_argument('--hours',type=float,default=20); p.add_argument('--steps',type=int,default=800); p.add_argument('--minutes',type=float,default=48)
    a=p.parse_args()
    if a.command=='catalog': continuous_catalog(Path(a.out),a.hours)
    elif a.command=='train': train(a)
    else: export(a)
