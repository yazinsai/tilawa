"""Bounded causal mean lesions of the shipped pre-head channels.

Fit/selection/audit partitions group training speakers. Only none takes select
lesions; confirmed scripted substitutions and correctly recognized words protect
existing distinctions. Practice inputs are accepted only by the diagnostic verb.
"""
import argparse
import hashlib
import json
from pathlib import Path

RANDOM_MASKS = 19


def ctc_scores(log_probs, ids):
    """Batched ordinary CTC forward sum with complete accepting states."""
    import numpy as np
    x = np.asarray(log_probs)
    if x.ndim == 2:
        x = x[None]
    batch, frames, _ = x.shape
    if not frames:
        return np.full(batch, -np.inf if ids else 0.)
    labels = np.array([0] + [p for v in ids for p in (v, 0)])
    prev = np.full((batch, len(labels)), -np.inf)
    prev[:, 0] = x[:, 0, 0]
    if ids:
        prev[:, 1] = x[:, 0, ids[0]]
    skip = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    for t in range(1, frames):
        nxt = prev.copy()
        nxt[:, 1:] = np.logaddexp(nxt[:, 1:], prev[:, :-1])
        nxt[:, 2:] = np.logaddexp(nxt[:, 2:], np.where(skip, prev[:, :-2], -np.inf))
        prev = nxt + x[:, t, labels]
    return np.logaddexp(prev[:, -1], prev[:, -2]) if ids else prev[:, 0]


def log_softmax(z):
    import numpy as np
    peak = z.max(-1, keepdims=True)
    return z - peak - np.log(np.exp(z - peak).sum(-1, keepdims=True))


def lesioned_logits(features, logits, weight, mean, masks):
    import numpy as np
    arms = []
    for mask in masks:
        arms.append(logits + (mean[mask] - features[:, mask]) @ weight[mask] if mask else logits)
    return np.array(arms)


def load_cache(path, meta):
    import numpy as np
    path = Path(path)
    rows = [json.loads(s) for s in (path / 'rows.jsonl').read_text().splitlines()]
    for r in rows:
        for name, width in [('features', meta['width']), ('logits', len(meta['tokens'])+1), ('logProbs', len(meta['tokens'])+1)]:
            r[name] = np.fromfile(path / f"{r['cache']}-{name}.f32", dtype=np.float32).reshape(r['frames'], width).astype(np.float64)
        # Effective head reconstruction, including dynamic quantizer zero points.
        np.testing.assert_allclose(r['features'] @ np.array(meta['effectiveWeight']) + meta['bias'], r['logits'], atol=3e-5, rtol=1e-5)
        np.testing.assert_allclose(log_softmax(r['logits']), r['logProbs'], atol=2e-6, rtol=1e-5)
    return rows


def ids(text, tokens):
    return [tokens.index(p)+1 for p in text]


def comparison(r, c, tokens, role):
    a, b = c['fromFrame'], c['toFrame']
    if not 0 <= a < b <= r['frames']:
        return None
    sequences = [ids(c[k], tokens) for k in ('written', 'rival')]
    if not all(sequences):
        return None
    scores = [float(ctc_scores(r['logProbs'][a:b], s)[0]) for s in sequences]
    if not all(__import__('math').isfinite(s) for s in scores):
        return None
    return dict(speaker=r.get('speaker', ''), sourceId=r['id'], candidate=c, role=role,
                features=r['features'][a:b], logits=r['logits'][a:b], sequences=sequences,
                originalMargin=scores[0]-scores[1])


def viterbi(lp, pronunciation):
    """Canonical placement used only for training preservation controls."""
    import numpy as np
    if not len(lp) or not pronunciation:
        return None
    labels = np.array([0] + [p for v in pronunciation for p in (v, 0)])
    prev = np.full(len(labels), -np.inf)
    prev[:2] = lp[0, labels[:2]]
    back = np.zeros((len(lp), len(labels)), dtype=np.int32)
    skip = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    for t in range(1, len(lp)):
        choices = np.stack([prev, np.r_[-np.inf,prev[:-1]], np.r_[-np.inf,-np.inf,np.where(skip,prev[:-2],-np.inf)]])
        step = choices.argmax(0)
        back[t] = np.arange(len(labels))-step
        prev = choices.max(0)+lp[t,labels]
    state = len(labels)-1 if prev[-1]>prev[-2] else len(labels)-2
    if not np.isfinite(prev[state]):
        return None
    path = []
    for t in range(len(lp)-1,-1,-1):
        path.append(state)
        state = back[t,state]
    return np.array(path[::-1])


def controls(rows, tokens):
    """Attested contrasts of complete training words already recognized correctly."""
    import numpy as np
    lexicon = sorted({w['pronunciation'] for r in rows if r['kind']=='none' for w in r['words']})
    out = []
    for r in rows:
        if r['kind']!='none' or not r['pronunciation']:
            continue
        path = viterbi(r['logProbs'], ids(r['pronunciation'], tokens))
        if path is None:
            continue
        offset = 0
        for w in r['words']:
            text = w['pronunciation']
            positions = np.where((path>=2*offset+1)&(path<=2*(offset+len(text))-1))[0]
            offset += len(text)
            if not len(positions):
                continue
            a, b = int(positions[0]), int(positions[-1])+1
            greedy, previous = [], 0
            for p in r['logProbs'][a:b].argmax(1):
                if p and p!=previous:
                    greedy.append(int(p))
                previous = p
            if greedy!=ids(text,tokens):
                continue
            rival = next((p for p in lexicon if len(p)==len(text) and sum(x!=y for x,y in zip(p,text))==1), None)
            if not rival:
                continue
            c = comparison(r,dict(word=w['word'],written=text,rival=rival,fromFrame=a,toFrame=b),tokens,'correct')
            if c and c['originalMargin']>0:
                out.append(c)
    return out


def margins(comparisons, meta, mean, masks):
    import numpy as np
    weight = np.array(meta['effectiveWeight'])
    out = []
    for c in comparisons:
        lp = log_softmax(lesioned_logits(c['features'], c['logits'], weight, mean, masks))
        out.append(ctc_scores(lp,c['sequences'][0])-ctc_scores(lp,c['sequences'][1]))
    return np.array(out).reshape(len(comparisons),len(masks))


def partition_summary(comparisons, values):
    import numpy as np
    false = np.array([c['role']=='false' for c in comparisons])
    original = np.array([c['originalMargin'] for c in comparisons])
    reversals = false[:,None] & (values>0)
    losses = (~false)[:,None] & ((values>0)!=(original[:,None]>0))
    return dict(reversals=reversals.sum(0).tolist(), preservationLosses=losses.sum(0).tolist())


def train(args):
    import numpy as np
    meta = json.loads(Path(args.metadata).read_text())
    rows = load_cache(args.cache,meta)
    excluded = set(json.loads((Path(args.cache)/'excluded-speakers.json').read_text()))
    if any(r.get('source')!='help-train' or r.get('split')!='train' or r.get('excluded') is not False or
           not r.get('speaker') or r['speaker'] in excluded for r in rows):
        raise ValueError('Only eligible help-train training speakers may select the mask')
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers)<3:
        raise ValueError('Three speaker-separated training partitions required')
    # Stable shuffled speaker grouping avoids adjacent source identifiers defining folds.
    speakers.sort(key=lambda s:hashlib.sha256(s.encode()).hexdigest())
    folds = {s:i%3 for i,s in enumerate(speakers)}
    fitting = [r for r in rows if folds[r['speaker']]==0]
    # The lesion value uses fitting recordings only; selection/audit remain unseen.
    mean_inputs = [r['features'] for r in fitting if r['kind']=='none' and r['frames']]
    if not mean_inputs:
        raise ValueError('No none-labelled fitting-speaker features for training mean')
    mean = np.concatenate(mean_inputs).mean(0)
    comparisons = controls(rows,meta['tokens'])
    for r in rows:
        for request in r['candidates']:
            role = None
            if r['kind']=='none' and not r.get('events') and not r.get('non_errors'):
                role = 'false'
            elif any(e.get('errorClass')=='wrong_word' and e.get('state')=='confirmed' and
                     e.get('location')==request['word'] for e in r.get('events',[])):
                # A confirmed accusation at the scripted location protects the shipped
                # distinction; the script is never imposed as a performed phone label.
                role = 'scripted'
            if role:
                c = comparison(r,request,meta['tokens'],role)
                if c and c['originalMargin']<0:
                    comparisons.append(c)
    group = {fold:[c for c in comparisons if folds[c['speaker']]==fold] for fold in range(3)}
    if not all(any(c['role']=='false' for c in group[f]) for f in (0,1)):
        raise ValueError('Fitting and selection speakers must both have original rival-preferring false comparisons')
    if not all(any(c['role']==role for c in comparisons) for role in ('correct','scripted')):
        raise ValueError('Correct-word and confirmed scripted-substitution preservation controls required')
    limit = max(1,meta['width']//32)
    mask, history = [], []
    for step in range(limit):
        candidates = [mask+[d] for d in range(meta['width']) if d not in mask]
        value = margins(group[0],meta,mean,candidates)
        summary = partition_summary(group[0],value)
        choices = [i for i in range(len(candidates)) if summary['preservationLosses'][i]==0]
        if not choices:
            break
        # Causal contribution means the full pronunciation likelihood is recomputed,
        # including softmax and the entire CTC boundary sum, for every channel.
        false = np.array([c['role']=='false' for c in group[0]])
        best = max(choices,key=lambda i:(summary['reversals'][i],float(value[false,i].mean()),-candidates[i][-1]))
        mask = candidates[best]
        selection = partition_summary(group[1],margins(group[1],meta,mean,[mask]))
        history.append(dict(mask=mask.copy(),fitting={k:v[best] for k,v in summary.items()},
                            selection={k:v[0] for k,v in selection.items()}))
        print(json.dumps(history[-1]),flush=True)
    valid = [r for r in history if r['selection']['preservationLosses']==0]
    selected = max(valid,key=lambda r:(r['selection']['reversals'],r['fitting']['reversals'],-len(r['mask']))) if valid else None
    mask = selected['mask'] if selected else []
    rng = np.random.default_rng(1729)
    random = [sorted(rng.choice(meta['width'],len(mask),replace=False).tolist()) for _ in range(RANDOM_MASKS)]
    audit = partition_summary(group[2],margins(group[2],meta,mean,[mask]+random))
    model = {**meta,'mean':mean.tolist(),'mask':mask}
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'feature-lesion-mask.json').write_text(json.dumps(model,allow_nan=False))
    report = dict(trainingRecordings=len(rows),speakers=speakers,excludedSpeakers=sorted(excluded),speakerFolds=folds,
        meanRecordings=len([r for r in fitting if r['kind']=='none']),limit=limit,history=history,selected=selected,
        counts={str(f):{role:sum(c['role']==role for c in group[f]) for role in ('false','correct','scripted')} for f in range(3)},
        audit=audit,randomMasks=random,
        maskSha256=hashlib.sha256((out/'feature-lesion-mask.json').read_bytes()).hexdigest(),
        selection='fit channel contributions on partition 0; select sparse mask on partition 1; report untouched partition 2')
    (out/'feature-lesion-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)


def diagnose(args):
    import numpy as np
    model = json.loads(Path(args.model).read_text())
    report = json.loads(Path(args.report).read_text())
    rows = load_cache(args.cache,model)
    masks = [model['mask']]+report['randomMasks']
    comparisons = [c for r in rows for request in r['candidates']
                   if (c:=comparison(r,request,model['tokens'],'diagnostic'))]
    values = margins(comparisons,model,np.array(model['mean']),masks)
    output = [dict(candidate=c['candidate'],originalMargin=c['originalMargin'],
                   lesionedMargin=float(v[0]),randomMargins=v[1:].tolist(),
                   reversal=bool(c['originalMargin']<0<v[0])) for c,v in zip(comparisons,values)]
    Path(args.out).write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(output),flush=True)


def bundle(args):
    """Keep the caller's complete primary bundle, adding only hashed lesion assets."""
    import shutil
    source, out = Path(args.source), Path(args.out)
    manifest = json.loads((source/'manifest.json').read_text())
    mask = json.loads(Path(args.mask).read_text())
    checksum = hashlib.sha256((source/manifest['file']).read_bytes()).hexdigest()
    if checksum != manifest['sha256'] or mask['primarySha256'] != checksum or not mask['mask']:
        raise ValueError('A selected nonempty mask tied to the original primary is required')
    if mask['tokens'] != manifest['tokens'] or mask['inventory'] != manifest['inventory']:
        raise ValueError('Primary alphabet mismatch')
    meta = json.loads((Path(args.extractor)/'feature-lesion-metadata.json').read_text())
    for key in ('primarySha256','tokens','inventory','effectiveWeight','featureOutput','logitOutput'):
        if mask[key] != meta[key]:
            raise ValueError('Mask and feature extractor differ')
    if source.resolve() == out.resolve():
        raise ValueError('Use a new bundle directory; never modify the primary source')
    shutil.copytree(source,out,dirs_exist_ok=True)
    shutil.copyfile(args.mask,out/'feature-lesion-mask.json')
    shutil.copyfile(Path(args.extractor)/'feature-lesion.onnx',out/'feature-lesion.onnx')
    files = []
    for name in ('feature-lesion-mask.json','feature-lesion.onnx'):
        data = (out/name).read_bytes()
        files.append(dict(file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (out/'feature-lesion-asset.json').write_text(json.dumps(dict(schema=1,files=files),indent=2))
    if (out/'manifest.json').read_bytes() != (source/'manifest.json').read_bytes():
        raise ValueError('Primary manifest changed')
    print(json.dumps(dict(bundle=str(out),files=files)),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='verb',required=True)
    t = sub.add_parser('train')
    for name in ('metadata','cache','out'):
        t.add_argument('--'+name,required=True)
    d = sub.add_parser('diagnose')
    for name in ('model','report','cache','out'):
        d.add_argument('--'+name,required=True)
    b = sub.add_parser('bundle')
    for name in ('source','extractor','mask','out'):
        b.add_argument('--'+name,required=True)
    args = p.parse_args()
    {'train':train,'diagnose':diagnose,'bundle':bundle}[args.verb](args)
