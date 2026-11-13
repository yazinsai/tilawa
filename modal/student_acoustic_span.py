"""Training-only coherent pronunciation vs same-phone fragment compositions.

All comparisons share positives, initialization, capacity and update budget. Phone
fragments come from another occurrence in the very same recording, so identity,
recording conditions and final durations match exactly. Same-occurrence positives
undergo identical split/interpolation/join operations. No practice data is read.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

SEED = 1729
EPOCHS = 24
BUDGET_SECONDS = 24 * 60 * 60


def partition(speaker):
    return int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) % 5


def resample(x, n):
    import numpy as np
    if len(x) == n:
        return x.copy()
    return np.stack([np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x[:, d])
                     for d in range(x.shape[1])], axis=1).astype(np.float32)


def phone_intervals(word):
    # CTC peaks anchor an equal midpoint split of uncertain blank intervals.
    bounds = word['phoneBounds']
    edges = [word['from']] + [(a[1] + b[0]) // 2 for a, b in zip(bounds, bounds[1:])] + [word['to']]
    return list(zip(edges, edges[1:]))


def compose_recording(row):
    """Each phone uses a different occurrence; every frame and phone is retained.

Donors must have comparable unwarped duration, and all belong to one take. Controls
resample their own fragments through the same intermediate length and back, exposing exactly
identical join positions and interpolation geometry. Three boundary splits integrate
alignment uncertainty instead of exposing a fixed synthetic cut to the classifier.
"""
    import numpy as np
    x = np.asarray(row['features'], dtype=np.float32)
    pool = {}
    for wi, w in enumerate(row['words']):
        for pi, (phone, (a, b)) in enumerate(zip(w['ids'], phone_intervals(w))):
            if b > a:
                pool.setdefault(phone, []).append((wi, pi, a, b))
    result = []
    for wi, w in enumerate(row['words']):
        intervals = phone_intervals(w)
        if len(w['ids']) < 2 or w['to'] - w['from'] > 100 or any(b <= a for a, b in intervals):
            continue
        donors = []
        for pi, (phone, (a, b)) in enumerate(zip(w['ids'], intervals)):
            options = [d for d in pool.get(phone, []) if d[:2] != (wi, pi) and
                       max(d[3]-d[2], b-a) / min(d[3]-d[2], b-a) <= 1.5]
            if not options:
                break
            donors.append(min(options, key=lambda d: (abs((d[3]-d[2])-(b-a)), d[:2])))
        if len(donors) != len(intervals) or len({d[:2] for d in donors}) < 2:
            continue
        coherent, fragments = [], []
        for jitter in (0, -1, 1):
            def shifted(parts):
                edges = [parts[0][0]] + [a + jitter for a, _ in parts[1:]] + [parts[-1][1]]
                if any(b <= a for a, b in zip(edges, edges[1:])):
                    return parts
                return list(zip(edges, edges[1:]))
            target = shifted(intervals)
            # Use the same bounded jitter on each donor's uncertain endpoint.
            donor_parts = [(max(0,a+jitter), min(len(x),b+jitter)) for _, _, a, b in donors]
            c, f = [], []
            for (a, b), (da, db) in zip(target, donor_parts):
                n, dn = b-a, db-da
                k = min(n, dn)
                c.append(resample(resample(x[a:b], k), n))
                f.append(resample(resample(x[da:db], k), n))
            coherent.append(np.concatenate(c)); fragments.append(np.concatenate(f))
        result.append(dict(ids=w['ids'], coherent=coherent, fragments=fragments,
                           take=row['id'], word=wi, donors=[d[:2] for d in donors]))
    return result


def make_head(vocabulary):
    import torch
    from torch import nn
    class Compatibility(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary, 16)
            self.hidden = nn.Linear(80, 32)
            self.output = nn.Linear(32, 1)
        def forward(self, x, ids):
            n, p = len(x), len(ids)
            # Adjacent observations interact nonlinearly; an independent emission
            # readout cannot detect cross-fragment acoustic incompatibility.
            delta = x - torch.cat((x[:1], x[:-1]))
            acoustic = torch.cat((x, delta), -1)[None].expand(p, n, 64)
            query = self.embedding(ids)[:, None].expand(p, n, 16)
            e = self.output(self.hidden(torch.cat((acoustic, query), -1)).tanh()).squeeze(-1) / n
            dp = torch.cat((e[:1, 0], torch.full((p-1,), -1e4, device=x.device)))
            for t in range(1, n):
                advance = torch.cat((torch.full((1,), -1e4, device=x.device), dp[:-1]))
                dp = torch.logaddexp(dp, advance) + e[:, t]
            return dp[-1] - (math.lgamma(n)-math.lgamma(p)-math.lgamma(n-p+1))
    return Compatibility()


def predict(head, x, ids):
    import torch
    n, p = len(x), len(ids)
    logits = [head(x[a:b], ids) for a,b in ((0,n),(1,n),(0,n-1)) if b-a >= p]
    return torch.logsumexp(torch.stack(logits), 0) - math.log(len(logits))


def serialize(head):
    return dict(embedding=head.embedding.weight.detach().tolist(), weight=head.hidden.weight.detach().tolist(),
                bias=head.hidden.bias.detach().tolist(), output=head.output.weight.detach()[0].tolist(),
                intercept=float(head.output.bias.detach()[0]))


def fit(args):
    import numpy as np
    import torch
    started = time.monotonic()
    torch.set_num_threads(1); torch.manual_seed(SEED)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    for r in rows:
        if (r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or
            not r.get('speaker') or r['speaker'] in excluded or r.get('kind') not in ('none', 'supported-substitution')):
            raise ValueError('Invalid positive training provenance')
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < 6:
        raise ValueError('Insufficient independent training speakers')
    fit_rows = [r for r in rows if partition(r['speaker']) < 3]
    features = np.concatenate([r['features'] for r in fit_rows]).astype(np.float32)
    center, scale = features.mean(0), np.maximum(features.std(0), .01)
    metadata.update(center=center.tolist(), scale=scale.tolist())
    partitions, audit = {}, {}
    for group in ('fit', 'select', 'audit'):
        chosen = [r for r in rows if ('fit' if partition(r['speaker'])<3 else 'select' if partition(r['speaker'])==3 else 'audit') == group]
        words = [w for r in chosen for w in compose_recording(r)][:300]
        if len(words) < 2:
            raise ValueError('Insufficient within-recording same-phone fragment matches in '+group)
        examples = {arm: [] for arm in ('contrast', 'control')}
        for i, w in enumerate(words):
            alternatives = [v for v in words if v['ids'] != w['ids'] and len(v['ids']) <= len(w['coherent'][0])]
            if not alternatives:
                continue
            rival = alternatives[i % len(alternatives)]['ids']
            ids = torch.tensor(w['ids'], dtype=torch.int64)
            rid = torch.tensor(rival, dtype=torch.int64)
            def tensor(v): return torch.tensor((v-center)/scale)
            # Shared ordinary pronunciation discrimination prevents an unconditional
            # coherence detector from winning regardless of the candidate phones.
            positive = [tensor(v) for v in w['coherent']]
            negative = [tensor(v) for v in w['fragments']]
            for arm in examples:
                examples[arm].append((positive, ids, 1.))
                examples[arm].append((positive, rid, 0.))
                # Identical capacity/data/update count control never sees fragments.
                examples[arm].append((negative if arm=='contrast' else positive,
                                      ids if arm=='contrast' else rid, 0.))
        if not examples['contrast']:
            raise ValueError('Empty pronunciation contrast partition')
        # Coherent scripted substitutions remain positive even when their rare phones
        # have no second occurrence from which a matched fragment can be made.
        preservation = 0
        for r in chosen:
            if r['kind'] != 'supported-substitution':
                continue
            for w in r['words'][:40]:
                x = np.asarray(r['features'][w['from']:w['to']], dtype=np.float32)
                if not w['ids'] or len(x)<len(w['ids']) or len(x)>100:
                    continue
                ids = torch.tensor(w['ids'], dtype=torch.int64)
                coherent = [torch.tensor((x-center)/scale)] * 3
                for arm in examples:
                    examples[arm].append((coherent,ids,1.))
                preservation += 1
        partitions[group] = examples
        audit[group] = dict(speakers=sorted({r['speaker'] for r in chosen}), words=len(words),
                            examples=len(examples['contrast']), substitutionPreservation=preservation, matched=[dict(take=w['take'], word=w['word'],
                            ids=w['ids'], donors=w['donors']) for w in words])
    initial = make_head(len(metadata['tokens'])+1).state_dict()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = dict(seed=SEED, epochs=EPOCHS, budgetSeconds=BUDGET_SECONDS, partitions=audit,
                  excludedSpeakers=sorted(excluded), inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                  codeSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), arms={})
    def loss_one(head, example):
        views, ids, y = example
        score = torch.logsumexp(torch.stack([predict(head, x, ids) for x in views]), 0) - math.log(len(views))
        return torch.nn.functional.binary_cross_entropy_with_logits(score, torch.tensor(y))
    for arm in ('contrast', 'control'):
        head = make_head(len(metadata['tokens'])+1); head.load_state_dict(initial)
        opt = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.01)
        data = partitions['fit'][arm]; rng = random.Random(SEED)
        best, loss_best, history = None, float('inf'), []
        for epoch in range(EPOCHS):
            if time.monotonic()-started > BUDGET_SECONDS:
                raise TimeoutError('One-day cached-feature training ceiling exceeded')
            order = list(range(len(data))); rng.shuffle(order); head.train()
            for offset in range(0,len(order),16):
                if time.monotonic()-started > BUDGET_SECONDS:
                    raise TimeoutError('One-day cached-feature training ceiling exceeded')
                opt.zero_grad(); batch = order[offset:offset+16]
                loss = torch.stack([loss_one(head,data[i]) for i in batch]).mean(); loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),5); opt.step()
            head.eval()
            with torch.no_grad():
                loss = float(torch.stack([loss_one(head,e) for e in partitions['select'][arm]]).mean())
            history.append(dict(epoch=epoch+1, selectionBCE=loss))
            if loss < loss_best:
                best=copy.deepcopy(head.state_dict()); loss_best=loss
            print(json.dumps(dict(arm=arm, **history[-1])),flush=True)
        head.load_state_dict(best); head.eval()
        with torch.no_grad():
            observed = [dict(loss=float(loss_one(head,e)),label=e[2]) for e in partitions['audit'][arm]]
            fixtures = [dict(features=(x.numpy()*scale+center).tolist(),ids=ids.tolist(),score=float(predict(head,x,ids)))
                        for views,ids,y in partitions['audit'][arm][:12] for x in views[:1]]
        (out/('acoustic-span-'+arm+'.json')).write_text(json.dumps(serialize(head),allow_nan=False))
        (out/('acoustic-span-'+arm+'-parity.json')).write_text(json.dumps(fixtures))
        report['arms'][arm]=dict(history=history,selectedEpoch=min(history,key=lambda r:r['selectionBCE'])['epoch'],audit=observed)
    (out/'acoustic-span.json').write_text(json.dumps(metadata,allow_nan=False))
    names=['acoustic-span.json','acoustic-span-features.onnx','acoustic-span-contrast.json','acoustic-span-control.json']
    assets={n:dict(bytes=(out/n).stat().st_size,sha256=hashlib.sha256((out/n).read_bytes()).hexdigest()) for n in names}
    (out/'acoustic-span-assets.json').write_text(json.dumps(assets))
    report['elapsedSeconds']=time.monotonic()-started
    (out/'acoustic-span-training-report.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('input','metadata','excluded-speakers','out'):p.add_argument('--'+name,required=True)
    fit(p.parse_args())
