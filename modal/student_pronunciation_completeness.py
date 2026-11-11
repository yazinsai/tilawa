"""Frozen-encoder completeness head: intact versus naturally contiguous truncated speech.

No practice labels or windows enter fitting. None-labelled help-train speakers
supply intact anchors; substitution recordings are preservation controls only.
Split by speaker before normalization, fitting or operating-point calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIMENSIONS = 32
EPOCHS = 24


def make_head(vocab):
    import torch
    from torch import nn

    class Completeness(nn.Module):
        def __init__(self):
            super().__init__()
            self.phone = nn.Embedding(vocab, 24)
            self.key = nn.Linear(32, 24)
            self.local = nn.Sequential(nn.Linear(56, 32), nn.Tanh(), nn.Linear(32, 1))
            self.finish = nn.Linear(2, 1)

        def forward(self, features, pronunciation):
            # No word identity, reference flag, duration scalar or expected text.
            # Position prior orders phone-conditioned acoustic attention, but does
            # not require exact forced boundaries on either complete or cut crops.
            query = self.phone(pronunciation)
            key = self.key(features)
            t = torch.arange(features.shape[1], device=features.device, dtype=features.dtype)
            p = torch.arange(pronunciation.shape[1], device=features.device, dtype=features.dtype)
            t = (t + .5) / features.shape[1]
            p = (p + .5) / pronunciation.shape[1]
            attention = (query @ key.transpose(1, 2) / (24 ** .5) - 12 * (p[:, None] - t[None, :]) ** 2).softmax(-1)
            pooled = attention @ features
            evidence = self.local(torch.cat((pooled, query), -1)).squeeze(-1)
            aggregate = torch.stack((evidence.mean(-1), evidence.min(-1).values), -1)
            return self.finish(aggregate).squeeze(-1).sigmoid()

    return Completeness()


def crop_views(features):
    """Same uncertain edge mixture as SDK, retaining the exact window as one view."""
    n = len(features)
    return [features, features[min(1, n-1):], features[:max(1, n-1)]]


def predict(model, features, ids):
    import torch
    return torch.stack([model(x[None], ids[None])[0] for x in crop_views(features)]).mean()


def examples(row):
    """Balance intact/cut pairs by pronunciation; preserve natural temporal continuity.

Each crop has three possible boundaries inferred on the intact take. Negatives
remove a first or last phone, or more, and half extend on the opposite edge into
neighbouring real speech to overlap the intact duration range. No waveform splice.
"""
    x = row['features']
    result = []
    for word in row['words']:
        a, b, ids = word['from'], word['to'], word['ids']
        boundaries = word['phoneBounds']
        if len(ids) < 2 or b-a < 4 or len(boundaries) != len(ids):
            continue
        for side in ('start', 'end'):
            k = max(1, len(ids)//3)
            cut = boundaries[k][0] if side == 'start' else boundaries[-k-1][1]
            if cut <= a or cut >= b:
                continue
            for extended in (False, True):
                # The incomplete crop retains no removed source samples.
                lo, hi = (cut, b) if side == 'start' else (a, cut)
                removed = (cut-a) if side == 'start' else (b-cut)
                if extended:
                    if side == 'start': hi = min(len(x), b+removed)
                    else: lo = max(0, a-removed)
                if hi-lo < 3: continue
                result.append(dict(ids=ids, x=x[lo:hi], y=0, speaker=row['speaker'], pronunciation=tuple(ids), extended=extended))
                # Matching complete anchors include the same context extension.
                pa, pb = a, b
                if extended:
                    if side == 'start': pb = min(len(x), b+removed)
                    else: pa = max(0, a-removed)
                result.append(dict(ids=ids, x=x[pa:pb], y=1, speaker=row['speaker'], pronunciation=tuple(ids), extended=extended))
    return result


def train(args):
    import numpy as np
    import onnxruntime as ort
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    out = Path(args.out)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    for r in rows:
        if r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Invalid training-speaker provenance')
    none = [r for r in rows if r['kind'] == 'none']
    speakers = sorted(set(r['speaker'] for r in none))
    if len(speakers) < 6:
        raise ValueError('Need six independent training speakers for fit, selection and calibration')
    groups = {s: i % 5 for i, s in enumerate(speakers)}
    # Three groups fit, one selects checkpoint, final group calibrates cutoff.
    fit_rows = [r for r in none if groups[r['speaker']] < 3]
    raw = np.concatenate([r['features'] for r in fit_rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    data = [e for r in none for e in examples(r)]
    for e in data:
        e['features'] = torch.tensor((np.asarray(e['x'],dtype=np.float32)-center)/scale)
        e['phones'] = torch.tensor(e['ids'], dtype=torch.int64)
    fitting = [e for e in data if groups[e['speaker']] < 3]
    selection = [e for e in data if groups[e['speaker']] == 3]
    calibration = [e for e in data if groups[e['speaker']] == 4]
    if not all((fitting, selection, calibration)):
        raise ValueError('Empty speaker-separated completeness partition')
    counts = {}
    for e in fitting: counts[e['pronunciation']] = counts.get(e['pronunciation'], 0)+1
    weights = np.array([1/counts[e['pronunciation']] for e in fitting]); weights /= weights.sum()
    model = make_head(len(metadata['tokens'])+1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    rng = np.random.default_rng(1729)
    best, best_loss, history = None, float('inf'), []
    for epoch in range(EPOCHS):
        model.train()
        for i in rng.choice(len(fitting), size=len(fitting), p=weights):
            e=fitting[i]
            optimizer.zero_grad()
            p=predict(model,e['features'],e['phones'])
            value=torch.nn.functional.binary_cross_entropy(p,torch.tensor(float(e['y'])))
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss=float(np.mean([float(torch.nn.functional.binary_cross_entropy(predict(model,e['features'],e['phones']),torch.tensor(float(e['y'])))) for e in selection]))
        history.append(dict(epoch=epoch+1,selectionBCE=loss))
        if loss < best_loss:
            best_loss=loss;best={k:v.detach().clone() for k,v in model.state_dict().items()}
        print(json.dumps(history[-1]),flush=True)
    model.load_state_dict(best);model.eval()
    with torch.no_grad():
        observed=[dict(probability=float(predict(model,e['features'],e['phones'])),label=e['y'],speaker=e['speaker'],extended=e['extended']) for e in calibration]
        controls=[]
        for r in rows:
            if r['kind']=='none' or groups.get(r['speaker']) != 4: continue
            for c in r.get('controls',[]):
                x=torch.tensor((np.asarray(c['features'],dtype=np.float32)-center)/scale)
                controls.append(float(predict(model,x,torch.tensor(c['rival'],dtype=torch.int64))))
    # Freeze strictly below every complete calibration anchor and genuine engine-
    # caught weak substitution control. Confidence must also identify incomplete
    # rather than merely prefer it to an even less probable alternative.
    preservation=[v['probability'] for v in observed if v['label']==1]+controls
    if not controls: raise ValueError('No held-training-speaker substitution preservation controls')
    cutoff=min(.5,min(preservation))
    if cutoff<=0: raise ValueError('No nonzero preservation operating point')
    metadata.update(center=center.tolist(),scale=scale.tolist(),cutoff=cutoff)
    out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/'completeness-head.pt')
    target=out/'completeness-head.onnx'
    torch.onnx.export(model,(torch.randn(1,20,32),torch.tensor([[1,2,3]])),str(target),opset_version=17,
        input_names=['features','pronunciation'],output_names=['complete_probability'],
        dynamic_axes={'features':{1:'frames'},'pronunciation':{1:'phones'}},dynamo=False)
    session=ort.InferenceSession(str(target),providers=['CPUExecutionProvider'])
    parity=[]
    for e in calibration[:20]:
        for x in crop_views(e['features']):
            ref=float(model(x[None],e['phones'][None])[0].detach())
            got=float(session.run(None,{'features':x[None].numpy(),'pronunciation':e['phones'][None].numpy()})[0][0])
            if abs(ref-got)>1e-5: raise ValueError('Completeness export parity failed')
            parity.append(abs(ref-got))
    (out/'pronunciation-completeness.json').write_text(json.dumps(metadata,allow_nan=False))
    names=['pronunciation-completeness.json','completeness-features.onnx','completeness-head.onnx']
    descriptor={n:dict(bytes=(out/n).stat().st_size,sha256=hashlib.sha256((out/n).read_bytes()).hexdigest()) for n in names}
    (out/'completeness-assets.json').write_text(json.dumps(descriptor))
    report=dict(speakers=speakers,groups=groups,recordings=len(none),fitting=len(fitting),selection=len(selection),calibration=len(calibration),
      controls=len(controls),cutoff=cutoff,calibrationRejected=sum(v['probability']<cutoff for v in observed if v['label']==0),
      calibrationIncomplete=sum(v['label']==0 for v in observed),history=history,parityMax=max(parity),
      excludedSpeakers=sorted(excluded),inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'completeness-training-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('input','metadata','excluded-speakers','out'): p.add_argument('--'+name,required=True)
    train(p.parse_args())
