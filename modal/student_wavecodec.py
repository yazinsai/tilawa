"""Small reconstruction-only VQ waveform codec, followed by a frozen phone scorer.

All fitting accepts only help-train training speakers. Codec fitting never reads
pronunciations, teacher targets or engine boundaries. Phone fitting uses only
none-labelled takes. Grouped selection is completed before target inspection.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from student_wavecodec_emissions import fit, boundary, emissions

WIDTH = 16
PATCH = 400
STEPS = 1200
MAX_SECONDS = 23 * 3600


class Codec(nn.Module):
    def __init__(self, codes):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(1, 8, 8, 4), nn.Tanh(),
            nn.Conv1d(8, 16, 8, 4), nn.Tanh(), nn.AdaptiveAvgPool1d(1))
        self.codebook = nn.Parameter(torch.randn(codes, WIDTH) * .1)
        self.decoder = nn.Sequential(nn.Linear(WIDTH, 64), nn.Tanh(), nn.Linear(64, PATCH))

    def encode(self, x):
        z = self.encoder(x).squeeze(-1)
        distance = (z[:, None] - self.codebook[None]).square().sum(-1)
        ids = distance.argmin(1)
        return z, self.codebook[ids], ids

    def forward(self, x):
        z, q, ids = self.encode(x)
        straight = z + (q-z).detach()
        return self.decoder(straight), (q-z.detach()).square().mean() + .25*(z-q.detach()).square().mean(), ids


def waveform(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, rate = sf.read(str(path), dtype='float32', always_2d=True)
    x = x.mean(1)
    if rate != 16000:
        import math
        d = math.gcd(rate, 16000)
        x = resample_poly(x, 16000//d, rate//d).astype(np.float32)
    if len(x) < PATCH or not np.isfinite(x).all():
        raise ValueError('Invalid training waveform')
    return x


def train_codec(waves, codes, seed, deadline):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Codec(codes)
    if sum(p.numel() for p in model.parameters()) > 100000:
        raise ValueError('Codec exceeds CPU experiment parameter bound')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []
    for step in range(STEPS):
        if time.monotonic() > deadline:
            raise TimeoutError('One-day CPU budget reached')
        # Uniform recording sampling prevents long recordings dominating reconstruction.
        batch = []
        for _ in range(64):
            x = waves[int(rng.integers(len(waves)))]['audio']
            start = int(rng.integers(len(x)-PATCH+1))
            batch.append(x[start:start+PATCH])
        target = torch.tensor(np.stack(batch)) * 10
        prediction, commitment, _ = model(target[:, None])
        reconstruction = (prediction-target).square().mean()
        loss = reconstruction + commitment
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 400 == 0:
            history.append(float(reconstruction.detach()))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, history


def observations(model, x, hop):
    # Left-aligned, unchanged 25 ms waveform windows; fixed gain retains quiet detail.
    patches = np.lib.stride_tricks.sliding_window_view(x, PATCH)[::hop].copy()
    result = []
    with torch.no_grad():
        for start in range(0, len(patches), 512):
            _, q, _ = model.encode(torch.tensor(patches[start:start+512,None])*10)
            result.append(q.numpy())
    return np.concatenate(result)


def read_training(args):
    root = Path(args.help).resolve()
    labels = [json.loads(s) for s in (root/'labels.jsonl').read_text().splitlines() if s]
    denied = set(json.loads(Path(args.excluded_speakers).read_text()))
    denied.update(r['speaker'] for r in labels if r['split'] != 'train')
    eligible = [r for r in labels if r['split']=='train' and r['speaker'] not in denied]
    if len({r['speaker'] for r in eligible}) < 3:
        raise ValueError('At least three eligible training speakers required')
    waves = []
    for r in eligible:
        path = (root/r['wav']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Training audio escapes help-train')
        waves.append(dict(id=r['id'], speaker=r['speaker'], audio=waveform(path)))
    none = {r['id']:r for r in eligible if r['kind']=='none' and not r['events'] and not r['non_errors']}
    # The native SDK extractor supplies canonical pronunciations and frozen pre-head
    # features only for none-labelled takes. They are never passed to train_codec.
    prepared = [json.loads(s) for s in Path(args.features).read_text().splitlines() if s]
    for r in prepared:
        if r['id'] not in none or r['speaker'] != none[r['id']]['speaker']:
            raise ValueError('Phone training contains an ineligible take')
    if set(r['id'] for r in prepared) != set(none):
        raise ValueError('Missing none-labelled training observations')
    return waves, prepared, denied


def paired_rows(codec, waves, prepared, hop, tokens):
    audio = {r['id']:r['audio'] for r in waves}
    rows = []
    for r in prepared:
        x = observations(codec, audio[r['id']], hop)
        c = np.array(r['control'], dtype=float)
        # Same observation grid/capacity; nearest native 40 ms encoder frame.
        n = min(len(x), int(len(c)*640/hop))
        control = c[np.minimum((np.arange(n)*hop//640),len(c)-1)]
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if n < len(ids):
            raise ValueError('Complete canonical pronunciation cannot fit recording')
        rows.append(dict(codec=x[:n],control=control,ids=ids,speaker=r['speaker']))
    return rows


def contrast_validation(arm, rows, key, seed):
    # Training-only evaluation: each correct speaker-held-out pronunciation versus
    # one deterministic phone replacement at every eligible interior position.
    rng = np.random.default_rng(seed)
    speakers = {}
    for r in rows:
        values = emissions(arm,(r[key]-arm['center'])/arm['scale'])
        correct = boundary(values,r['ids'])
        rivals = []
        for at in range(len(r['ids'])):
            ids = r['ids'].copy()
            other = int(rng.integers(1,len(arm['phones'])))
            if other == ids[at]:
                other = other % (len(arm['phones'])-1) + 1
            ids[at] = other
            rivals.append(boundary(values,ids))
        # Bounded preference loss compares representations without density units.
        loss = float(np.mean(np.logaddexp(0,np.array(rivals)-correct)))
        speakers.setdefault(r['speaker'],[]).append(loss)
    return float(np.mean([np.mean(v) for v in speakers.values()]))


def main(args):
    torch.set_num_threads(2)
    deadline = time.monotonic()+MAX_SECONDS
    waves, prepared, denied = read_training(args)
    meta = json.loads(Path(args.metadata).read_text())
    speakers = sorted({r['speaker'] for r in waves},key=lambda s:hashlib.sha256(s.encode()).digest())
    folds = {s:i%3 for i,s in enumerate(speakers)}
    if any(not any(folds[r['speaker']]==f for r in prepared) for f in range(3)):
        raise ValueError('Every speaker fold needs none-labelled examples')
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    results = []
    for codes in (32,64):
        for fold in range(3):
            training = [r for r in waves if folds[r['speaker']]!=fold]
            codec, history = train_codec(training,codes,1729+fold,deadline)
            for hop in (160,320):
                rows = paired_rows(codec,waves,prepared,hop,meta['tokens'])
                train = [r for r in rows if folds[r['speaker']]!=fold]
                valid = [r for r in rows if folds[r['speaker']]==fold]
                for strength in (4,16,64):
                    for key in ('codec','control'):
                        if time.monotonic()>deadline: raise TimeoutError('One-day CPU budget reached')
                        arm = fit(train,key,len(meta['tokens'])+1,strength)
                        loss = contrast_validation(arm,valid,key,2718)
                        item = dict(codes=codes,hop=hop,fold=fold,strength=strength,arm=key,loss=loss)
                        results.append(item); print(json.dumps(item),flush=True)
    choices = []
    for codes in (32,64):
        for hop in (160,320):
            selected = {}
            for key in ('codec','control'):
                scores = {s:float(np.mean([r['loss'] for r in results if r['codes']==codes and r['hop']==hop and r['strength']==s and r['arm']==key])) for s in (4,16,64)}
                selected[key] = min(scores,key=scores.get)
            loss = np.mean([r['loss'] for r in results if r['codes']==codes and r['hop']==hop and r['strength']==selected[r['arm']]])
            choices.append(dict(codes=codes,hop=hop,strength=selected,loss=float(loss)))
    chosen = min(choices,key=lambda r:r['loss'])
    codec, history = train_codec(waves,chosen['codes'],1729,deadline)
    rows = paired_rows(codec,waves,prepared,chosen['hop'],meta['tokens'])
    state = {k:v.tolist() for k,v in codec.state_dict().items() if k.startswith('encoder.') or k=='codebook'}
    model = dict(meta,schema=1,dimensions=WIDTH,hop=chosen['hop'],patch=PATCH,weights=state,
        codec=fit(rows,'codec',len(meta['tokens'])+1,chosen['strength']['codec']),
        control=fit(rows,'control',len(meta['tokens'])+1,chosen['strength']['control']))
    target = out/'wavecodec-emissions.json'; target.write_text(json.dumps(model,allow_nan=False))
    torch.save(codec.state_dict(),out/'wavecodec-reconstruction.pt')
    report = dict(waveformRecordings=len(waves),noneRecordings=len(rows),speakerFolds=folds,
        excludedSpeakers=sorted(denied),selection=results,chosen=chosen,reconstructionHistory=history,
        parameters=sum(p.numel() for p in codec.parameters()),steps=STEPS,cpuThreads=2,
        inputSha256=hashlib.sha256(Path(args.features).read_bytes()).hexdigest())
    (out/'wavecodec-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    files = [target,out/'wavecodec-control.onnx']
    descriptor = dict(schema=1,files=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (out/'wavecodec-asset.json').write_text(json.dumps(descriptor))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('help','features','metadata','excluded-speakers','out'):
        p.add_argument('--'+name,required=True)
    main(p.parse_args())
