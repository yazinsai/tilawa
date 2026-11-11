"""Independent overlapping-band CTC ear and an exactly parameter-matched early-mixing control.

The only ablation is a fixed orthonormal broadband frequency mix before identical
encoders. Corruption is applied before that mix, identically in the two arms.
No practice labels, locations, audio or posteriors enter fitting or selection.
Run directly in the authorized training image; no Modal client is required.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from student import TOKENS, phoneme_ids, spliced_target, exclusions, mix_noise, ctc_collapse, edit_distance

BANDS = ((0,32),(16,48),(32,64),(48,80))
DILATIONS = (1,2,4,8,16,1)


def frequency_mix():
    # Orthonormal DCT-II. Each control encoder sees contributions from all 80 bins.
    k = torch.arange(80)[:,None]
    n = torch.arange(80)[None,:]
    m = torch.cos(math.pi/80*(n+.5)*k) * math.sqrt(2/80)
    m[0] /= math.sqrt(2)
    return m


class Residual(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation)
        self.point = nn.Linear(width,width)

    def forward(self, x):
        z = self.norm(x)
        z = F.silu(self.conv(z.transpose(1,2)).transpose(1,2))
        return x + .5*self.point(z)


class BandEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embed = nn.Conv1d(32,width,9,stride=4)
        self.layers = nn.Sequential(*(Residual(width,d) for d in DILATIONS))
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(self.layers(F.silu(self.embed(x.transpose(1,2)).transpose(1,2))))


class SubbandEar(nn.Module):
    def __init__(self, width=128, early_mix=False):
        super().__init__()
        self.early_mix = early_mix
        self.register_buffer('mix', frequency_mix())
        self.register_buffer('center',torch.zeros(80))
        self.register_buffer('scale',torch.ones(80))
        self.encoders = nn.ModuleList(BandEncoder(width) for _ in BANDS)
        self.reliability = nn.ModuleList(nn.Linear(width,1) for _ in BANDS)
        self.phone = nn.Linear(width,len(TOKENS)+1)

    def forward(self, x):
        x = (x-self.center)/self.scale
        if self.early_mix:
            x = x @ self.mix.T
        # No encoder, normalization or reliability estimator receives another band's features.
        z = [encoder(x[:,:,a:b]) for encoder,(a,b) in zip(self.encoders,BANDS)]
        r = torch.cat([layer(v) for layer,v in zip(self.reliability,z)],dim=-1)
        weights = r.softmax(-1)
        fused = sum(v*weights[:,:,i:i+1] for i,v in enumerate(z))
        return self.phone(fused).log_softmax(-1), r.log_softmax(-1)


def corrupt(features, rng):
    """Locally independent band views; 25% retain the unchanged shipped-augmentation input.

    The known mask supplies reliability supervision, never phone or word labels.
    All corruption occurs in original-frequency space before control mixing.
    """
    x = features.clone()
    reliable = torch.ones_like(x)
    if rng.random() < .25:
        return x,reliable
    for lo,hi in BANDS:
        if rng.random() < .5:
            a = int(rng.integers(max(1,len(x)//2)))
            b = int(rng.integers(a+1,len(x)+1))
            # Independent replacement noise, not another word's acoustic evidence.
            width = hi-lo
            noise = torch.from_numpy(rng.normal(0,1,(b-a,width)).astype('float32'))
            x[a:b,lo:hi] = x[a:b,lo:hi].mean(0) + noise*x[a:b,lo:hi].std(0).clamp_min(1)
            reliable[a:b,lo:hi] = 0
    return x,reliable


def reliability_target(mask, mixed=False):
    if mixed:
        mask = mask @ frequency_mix().to(mask).square().T
    per_band = torch.stack([mask[:,:,a:b].mean(-1) for a,b in BANDS],-1)
    # Same kernel/stride as embedding: physical reliability over the samples seen at each frame.
    q = F.avg_pool1d(per_band.transpose(1,2),9,stride=4).transpose(1,2)
    q = q + .05  # all-corrupt views retain uncertainty, never a divide-by-zero target
    return q/q.sum(-1,keepdim=True)


def source_partition(metadata, gold):
    # The existing s1 manifest is speaker-cluster split. Exclude entire clusters
    # containing either a non-training clip or any gold clip, not just those rows.
    held = {r['cluster'] for r in metadata if r['cluster'] is not None and
        (r['split'] != 'train' or (r['shard'],r['row']) in gold)}
    selected = [r for r in metadata if r['cluster'] is not None and r['cluster'] not in held and
        r['split']=='train' and r['bucket']=='clean-match' and (r['shard'],r['row']) not in gold and
        16000 <= r['n_samples'] <= 30*16000]
    return selected,held


def load_sources(max_hours, seed):
    import pyarrow.parquet as pq
    gold = exclusions(Path('docs/benchmarks/gold-per-v1-exclude.txt').read_text())
    metadata = []
    for file in sorted(Path('/student/s1').glob('*.parquet')):
        for batch in pq.ParquetFile(file).iter_batches(columns=['shard','row','split','bucket','cluster','n_samples']):
            metadata.extend({**r,'file':file} for r in batch.to_pylist())
    selected,held_clusters = source_partition(metadata,gold)
    eligible = [(r['file'],r['shard'],r['row'],r['n_samples']) for r in selected]
    random.Random(seed).shuffle(eligible)
    chosen, samples = {},0
    for file,shard,row,n in eligible:
        chosen.setdefault(file,set()).add((shard,row)); samples += n
        if samples>=max_hours*3600*16000: break
    result = []
    for file,keys in chosen.items():
        for batch in pq.ParquetFile(file).iter_batches():
            for r in batch.to_pylist():
                if (r['shard'],r['row']) not in keys: continue
                result.append(dict(source='s1',flac=r['flac'],n=r['n_samples'],
                    target=spliced_target(r['ref'],r['hyp']),ids=r['top5_ids'],p=r['top5_p']))
    root = Path('/lab/data/help-train')
    labels = [json.loads(s) for s in (root/'labels.jsonl').read_text().splitlines() if s]
    excluded = {r['speaker'] for r in labels if r.get('split') not in (None,'train')}
    ayahs = {f"{a['s']}:{a['a']}":a['p'] for a in json.loads(Path('packages/core/assets/ref-v2.json').read_text())['ayahs']}
    help_rows = []
    import soundfile as sf
    from per_breakdown import collapse
    for r in labels:
        if r['kind']!='none' or not r.get('speaker') or r['speaker'] in excluded or r.get('events') or r.get('non_errors'): continue
        wav = (root/r['wav']).resolve()
        if not wav.is_relative_to(root): raise ValueError('Training audio outside help-train')
        info = sf.info(wav)
        if info.samplerate!=16000: raise ValueError('Training WAV must be 16 kHz')
        help_rows.append(dict(source='help-none',speaker=r['speaker'],wav=str(wav),n=info.frames,
            target=collapse(''.join(ayahs[a] for a in r['ayahs']))))
    speakers = sorted({r['speaker'] for r in help_rows})
    if len(speakers)<3: raise ValueError('Need three authorized none-labelled training speakers')
    # Deterministic grouped validation from the authorized training speakers, fixed before practice use.
    held = set(sorted(speakers,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:max(1,len(speakers)//5)])
    train_help = [r for r in help_rows if r['speaker'] not in held]
    valid = [r for r in help_rows if r['speaker'] in held]
    if not train_help or not valid or not result: raise ValueError('Empty authorized training partition')
    return result,train_help,valid,dict(s1Recordings=len(result),s1Hours=samples/16000/3600,
        s1ExcludedClusters=len(held_clusters),s1UnclusteredExcluded=sum(r['cluster'] is None for r in metadata),
        helpTrainingRecordings=len(train_help),helpValidationRecordings=len(valid),
        helpTrainingSpeakers=sorted(set(speakers)-held),helpValidationSpeakers=sorted(held),excludedSpeakers=sorted(excluded))


class Dataset:
    def __init__(self, rows, train, seed):
        self.rows,self.train,self.seed = rows,train,seed
        self.rng = None
        self.bank = {}
        if train:
            for name in ('noise','rir'):
                files = sorted(Path('/student/aug').glob(f'{name}*.npz'))
                if not files: raise ValueError(f'Missing shipped {name} augmentation bank')
                self.bank[name] = np.load(files[0])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        import soundfile as sf
        import torchaudio
        from scipy.signal import fftconvolve
        r = self.rows[i]
        if self.rng is None: self.rng = np.random.default_rng(torch.initial_seed() % 2**32)
        rng = self.rng
        wave,sr = sf.read(r['wav'] if 'wav' in r else io.BytesIO(r['flac']),dtype='float32')
        if sr!=16000: raise ValueError('Non-16 kHz source')
        if wave.ndim>1: wave=wave.mean(-1)
        apply_chain = self.train and rng.random() >= .25
        if apply_chain:
            def pick(name):
                bank=self.bank[name]; j=int(rng.integers(len(bank['offsets'])-1))
                return bank['data'][bank['offsets'][j]:bank['offsets'][j+1]].astype('float32')
            if rng.random()<.2:
                rir=pick('rir'); out=fftconvolve(wave,rir[int(np.argmax(np.abs(rir))):])[:len(wave)]
                wave=(out*np.sqrt(np.mean(wave**2))/max(1e-8,np.sqrt(np.mean(out**2)))).astype('float32')
            if rng.random()<.5: wave=mix_noise(wave,pick('noise'),rng.uniform(5,25),rng)
            wave=wave*10**(rng.uniform(-10,6)/20)
        x=torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave))[None],
            num_mel_bins=80,sample_frequency=16000,dither=0,energy_floor=1e-10,high_freq=-400)
        if apply_chain:
            # Preserve the shipped SpecAugment as well as extra independently corrupted bands.
            for _ in range(2):
                w=int(rng.integers(28)); a=int(rng.integers(81-w)); x[:,a:a+w]=0
            for _ in range(int(rng.integers(11))):
                w=int(rng.integers(max(1,int(len(x)*.05))+1)); a=int(rng.integers(max(1,len(x)-w)))
                x[a:a+w]=0
            x,mask=corrupt(x,rng)
        else: mask=torch.ones_like(x)
        ids=torch.from_numpy(np.frombuffer(r.get('ids',b''),np.uint8).astype('int64').reshape(-1,5))
        p=torch.from_numpy(np.frombuffer(r.get('p',b''),np.float16).astype('float32').reshape(-1,5))
        return dict(x=x,mask=mask,target=torch.tensor(phoneme_ids(r['target'])),ids=ids,p=p)


def collate(items):
    from torch.nn.utils.rnn import pad_sequence as pad
    return dict(x=pad([r['x'] for r in items],batch_first=True,padding_value=-23.0259),
        mask=pad([r['mask'] for r in items],batch_first=True),
        lengths=torch.tensor([max(0,(len(r['x'])-9)//4+1) for r in items]),
        targets=torch.cat([r['target'] for r in items]),target_lengths=torch.tensor([len(r['target']) for r in items]),
        ids=pad([r['ids'] for r in items],batch_first=True),p=pad([r['p'] for r in items],batch_first=True),
        teacher_lengths=torch.tensor([len(r['ids']) for r in items]))


def loss(model,batch,device):
    b={k:v.to(device) for k,v in batch.items()}
    with torch.autocast('cuda',dtype=torch.bfloat16): lp,r=model(b['x'])
    lp,r=lp.float(),r.float()
    t=torch.arange(lp.shape[1],device=device)[None,:]
    valid=t<b['lengths'][:,None]
    # CTC must be feasible; zero_infinity is not allowed to silently hide bad training transcripts.
    ctc=F.ctc_loss(lp.transpose(0,1),b['targets'],b['lengths'],b['target_lengths'],reduction='sum',zero_infinity=False)/valid.sum()
    q=reliability_target(b['mask'],model.early_mix)
    rel=(-(q*r).sum(-1)*valid).sum()/valid.sum()
    kd=lp.sum()*0
    if b['ids'].shape[1]:
        idx=torch.minimum(t.expand(len(lp),-1),(b['teacher_lengths']-1).clamp_min(0)[:,None])
        ids=b['ids'].gather(1,idx[:,:,None].expand(-1,-1,5))
        p=b['p'].gather(1,idx[:,:,None].expand(-1,-1,5)); p=p/p.sum(-1,keepdim=True).clamp_min(1e-6)
        use=valid & (b['teacher_lengths'][:,None]>0)
        kd=(-(p*lp.gather(-1,ids)).sum(-1)*use).sum()/use.sum().clamp_min(1)
    total=.5*ctc+kd+.2*rel
    return total,{'ctc':float(ctc.detach()),'kd':float(kd.detach()),'reliability':float(rel.detach())}


def train(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    source,help_rows,valid,provenance=load_sources(args.hours,args.seed)
    (out/'training-provenance.json').write_text(json.dumps(provenance,indent=2))
    # Each epoch retains source diversity and gives the small real-phone set15% of examples.
    train_rows=source+help_rows*max(1,int(len(source)*.15/.85/max(1,len(help_rows))))
    loader=torch.utils.data.DataLoader(Dataset(train_rows,True,args.seed),batch_size=args.batch,
        shuffle=True,num_workers=args.workers,collate_fn=collate,persistent_workers=args.workers>0)
    validation=torch.utils.data.DataLoader(Dataset(valid,False,args.seed),batch_size=4,num_workers=2,collate_fn=collate)
    models=[SubbandEar(args.width,False).cuda(),SubbandEar(args.width,True).cuda()]
    models[1].load_state_dict(models[0].state_dict())
    optim=[torch.optim.AdamW(m.parameters(),lr=.0008,weight_decay=.01) for m in models]
    start,step,best=time.monotonic(),0,float('inf')
    if args.resume and (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cuda',weights_only=False)
        for m,s,o,opt in zip(models,ck['models'],optim,ck['optim']):m.load_state_dict(s);o.load_state_dict(opt)
        step,best=ck['step'],ck['best']
    count=[sum(p.numel() for p in m.parameters()) for m in models]
    if count[0]!=count[1]:raise ValueError('Unmatched learned capacity')
    print(json.dumps({'parameters':count,'provenance':provenance}),flush=True)
    def save(name):
        torch.save(dict(models=[m.state_dict() for m in models],optim=[o.state_dict() for o in optim],
            width=args.width,step=step,best=best,parameters=count,provenance=provenance),out/name)
    def evaluate():
        totals=[0.,0.];n=0;pers=[[],[]]
        with torch.no_grad():
            for m in models:m.eval()
            for batch in validation:
                for a,m in enumerate(models):
                    value,_=loss(m,batch,'cuda');totals[a]+=float(value)*len(batch['lengths'])
                    lp,_=m(batch['x'].cuda()); pos=0
                    for i,l in enumerate(batch['lengths']):
                        nt=int(batch['target_lengths'][i]);ref=batch['targets'][pos:pos+nt].tolist();pos+=nt
                        hyp=ctc_collapse(lp[i,:int(l)].argmax(-1).tolist())
                        pers[a].append((edit_distance(hyp,ref),len(ref)))
                n+=len(batch['lengths'])
            for m in models:m.train()
        return [v/n for v in totals],[sum(e for e,l in p)/max(1,sum(l for e,l in p)) for p in pers]
    while step<args.steps and time.monotonic()-start<args.minutes*60:
        for batch in loader:
            metrics=[]
            for m,o in zip(models,optim):
                o.zero_grad(set_to_none=True);value,detail=loss(m,batch,'cuda')
                if not torch.isfinite(value):raise ValueError(f'Nonfinite training loss at {step}: {detail}')
                value.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);o.step();metrics.append(detail)
            step+=1
            if step%20==0:print(json.dumps({'step':step,'seconds':time.monotonic()-start,'arms':metrics}),flush=True)
            if step%args.eval_every==0:
                scores,per=evaluate();value=sum(scores)/2
                if value<best:best=value;save('best.pt')
                save('last.pt')
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps({'step':step,'loss':scores,'per':per,'selected':value==best})+'\n')
                print(json.dumps({'validationStep':step,'loss':scores,'per':per,'best':best}),flush=True)
            if step>=args.steps or time.monotonic()-start>=args.minutes*60:break
    scores,per=evaluate()
    if sum(scores)/2<best:best=sum(scores)/2;save('best.pt')
    save('last.pt')
    print(json.dumps({'finished':step,'validationLoss':scores,'validationPer':per,'best':best}),flush=True)


def export(args):
    import onnxruntime as ort
    import shutil
    ck=torch.load(Path(args.out)/'best.pt',map_location='cpu',weights_only=False)
    bundle=Path(args.bundle);bundle.mkdir(parents=True,exist_ok=True)
    primary=Path(args.primary)
    for file in primary.iterdir():
        if file.is_file():shutil.copy2(file,bundle/file.name)
    manifest=json.loads((primary/'manifest.json').read_text())
    if manifest['tokens']!=TOKENS:raise ValueError('Inventory mismatch')
    if hashlib.sha256((primary/manifest['file']).read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Invalid primary hash')
    x=torch.randn(1,409,80)
    class Output(nn.Module):
        def __init__(self,m):super().__init__();self.model=m
        def forward(self,x):return self.model(x)[0]
    comparisons=[]
    for a,name in enumerate(('subband-fused.onnx','subband-control.onnx')):
        model=SubbandEar(ck['width'],a==1);model.load_state_dict(ck['models'][a]);model.eval()
        torch.onnx.export(Output(model),x,bundle/name,input_names=['x'],output_names=['log_probs'],
            dynamic_axes={'x':{1:'input_frames'},'log_probs':{1:'output_frames'}},opset_version=17,dynamo=False)
        session=ort.InferenceSession(str(bundle/name),providers=['CPUExecutionProvider'])
        for length in (77,409,801):
            test=torch.randn(1,length,80)
            with torch.no_grad():expected=model(test)[0].numpy()
            observed=session.run(None,{'x':test.numpy()})[0]
            err=float(np.max(np.abs(expected-observed)))
            if err>.0001 or observed.shape[1]!=(length-9)//4+1:raise ValueError(f'ONNX mismatch:{name}:{err}')
            comparisons.append({'file':name,'frames':length,'maxError':err})
    cfg=dict(schema=1,inventory=manifest['inventory'],tokens=TOKENS,primarySha256=manifest['sha256'],
        mels=80,stride=4,kernel=9,bands=BANDS,learnedParameters=ck['parameters'][0],controlLearnedParameters=ck['parameters'][1])
    (bundle/'subband-ear-config.json').write_text(json.dumps(cfg))
    files=[]
    for name in ('subband-ear-config.json','subband-fused.onnx','subband-control.onnx'):
        raw=(bundle/name).read_bytes();files.append(dict(file=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    (bundle/'subband-ear-asset.json').write_text(json.dumps(dict(schema=1,files=files),indent=2))
    (Path(args.out)/'export-validation.json').write_text(json.dumps(comparisons,indent=2))
    print(json.dumps({'bundle':str(bundle),'step':ck['step'],'files':files,'parity':comparisons}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','export'])
    p.add_argument('--out',required=True);p.add_argument('--hours',type=float,default=30)
    p.add_argument('--width',type=int,default=128);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--steps',type=int,default=12000);p.add_argument('--minutes',type=int,default=50)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--seed',type=int,default=731);p.add_argument('--resume',action='store_true')
    p.add_argument('--primary');p.add_argument('--bundle');args=p.parse_args()
    if args.command=='train':train(args)
    else:export(args)
