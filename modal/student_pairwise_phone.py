"""Balanced nonlinear phone tournaments on frozen deployment encoder features.

Only none-labelled training speakers supply soft acoustic supervision. Confusion
mining, architecture, regularization and preservation decisions use training-only
speaker folds; scripts never become assumed performed phone transcripts.
"""
import argparse
import hashlib
import json
from pathlib import Path
from student_phone_occupancy import occupancies, sequence_score

DIMENSIONS, EPOCHS = 32, 18
REGULARIZATION = (.0001, .01)
CONSONANTS = set('ءبتثجحخدذرزسشصضطظعغفقكلمنهوي')


def mine_pairs(rows, tokens):
    import numpy as np
    vocab = len(tokens)+1
    confusion = np.zeros((vocab, vocab))
    mass = np.zeros(vocab)
    speakers = [set() for _ in range(vocab)]
    for r in rows:
        y = r['targets']
        # Each recording has equal influence, independent of length.
        decoded = np.zeros_like(y)
        decoded[np.arange(len(y)), np.asarray(r['logProbs']).argmax(1)] = 1
        confusion += y.T @ decoded / len(y)
        mass += y.mean(0)
        for i in range(1, vocab):
            if y[:, i].sum() >= 1:
                speakers[i].add(r['speaker'])
    eligible = [i for i in range(1, vocab) if tokens[i-1] in CONSONANTS and len(speakers[i]) >= 2]
    pairs = set()
    for i in eligible:
        rivals = sorted((j for j in eligible if j != i and confusion[i,j]+confusion[j,i] > 0), key=lambda j:
            (-(confusion[i,j]/mass[i]+confusion[j,i]/mass[j]), j))[:2]
        pairs.update(tuple(sorted((i,j))) for j in rivals)
    if not pairs:
        raise ValueError('No training-derived consonant contrasts with independent speakers')
    return sorted(pairs)


def architecture(n_pairs, vocab):
    """Exactly equal trainable parameter counts; no dummy unused parameters.

    Binary hidden widths may differ by one. If integer parity requires it, the
    final binary output uses no bias. The dense multiclass control remains a
    conventional fully connected MLP, using the identical input representation.
    """
    import math
    unit = DIMENSIONS+2
    for biases in (n_pairs, n_pairs-1):
        for h in range(max(8, math.ceil(n_pairs*4*unit/(DIMENSIONS+1+vocab))), 1000):
            budget = (DIMENSIONS+1+vocab)*h+vocab
            if (budget-biases) % unit == 0:
                total = (budget-biases)//unit
                if total >= 4*n_pairs:
                    widths = [total//n_pairs + (i < total % n_pairs) for i in range(n_pairs)]
                    return widths, biases, h, budget
    raise ValueError('No equal-parameter architecture')


def coupled_logits(control, pair_logits, pairs):
    """Least-squares Bradley-Terry log odds within each connected tournament.

    Sparse cycles need not be consistent. Solve their graph Laplacian with a
    zero-sum gauge, normalize within each component, and preserve its control
    probability mass. Unpaired classes and blank remain control probabilities.
    """
    import numpy as np
    result = np.array(control, copy=True)
    unseen = set(i for pair in pairs for i in pair)
    while unseen:
        group = {min(unseen)}
        while True:
            expanded = group | {v for a,b in pairs if a in group or b in group for v in (a,b)}
            if expanded == group:
                break
            group = expanded
        unseen -= group
        ids = sorted(group)
        edges = [(k,a,b) for k,(a,b) in enumerate(pairs) if a in group]
        incidence = np.zeros((len(edges), len(ids)))
        for e,(_,a,b) in enumerate(edges):
            incidence[e,ids.index(a)], incidence[e,ids.index(b)] = 1,-1
        inverse = np.linalg.pinv(incidence)
        scores = pair_logits[:,[k for k,_,_ in edges]] @ inverse.T
        normalized = scores - np.logaddexp.reduce(scores, axis=1)[:,None]
        mass = np.logaddexp.reduce(control[:,ids], axis=1)[:,None]
        result[:,ids] = normalized + mass
    return result


def fit(rows, tokens, reg):
    import numpy as np
    import torch
    torch.manual_seed(1729)
    pairs = mine_pairs(rows, tokens)
    widths, biases, hidden, count = architecture(len(pairs), len(tokens)+1)
    raw = np.concatenate([r['features'] for r in rows]).astype('float32')
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    x = torch.tensor((raw-center)/scale)
    y = np.concatenate([r['targets'] for r in rows])
    record_weights = np.concatenate([np.full(len(r['features']), 1/len(r['features'])) for r in rows])
    record_weights /= record_weights.sum()
    mass = (record_weights[:,None]*y).sum(0)
    control = torch.nn.Sequential(torch.nn.Linear(DIMENSIONS, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, len(tokens)+1))
    binaries = [torch.nn.Sequential(torch.nn.Linear(DIMENSIONS,w),torch.nn.Tanh(),torch.nn.Linear(w,1,bias=i<biases)) for i,w in enumerate(widths)]
    if sum(p.numel() for p in control.parameters()) != sum(p.numel() for h in binaries for p in h.parameters()):
        raise ValueError('Control capacity mismatch')
    training = []
    cy = y / np.maximum(mass, 1e-12)
    cw = cy.sum(1)*record_weights
    training.append((control, cy/cy.sum(1,keepdims=True), cw/cw.sum()))
    for h,(a,b) in zip(binaries,pairs):
        balanced = y[:,[a,b]] / mass[[a,b]]
        weight = balanced.sum(1)*record_weights
        target = balanced[:,0]/np.maximum(balanced.sum(1),1e-12)
        training.append((h, target[:,None], weight/weight.sum()))
    rng = np.random.default_rng(1729)
    steps = max(1,len(x)//256)
    for h, targets, weight in training:
        opt = torch.optim.AdamW(h.parameters(),lr=.002,weight_decay=reg)
        target = torch.tensor(targets,dtype=torch.float32)
        for _ in range(EPOCHS):
            for _ in range(steps):
                batch = rng.choice(len(x),256,p=weight)
                opt.zero_grad()
                logits = h(x[batch])
                loss = -(target[batch]*logits.log_softmax(-1)).sum(-1).mean() if h is control else torch.nn.functional.binary_cross_entropy_with_logits(logits,target[batch])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(),5.)
                opt.step()
        h.eval()
    return dict(control=control,binaries=binaries,pairs=pairs,center=center,scale=scale,parameters=count)


def outputs(fitted, features):
    import numpy as np
    import torch
    with torch.no_grad():
        x = torch.tensor((np.asarray(features,dtype='float32')-fitted['center'])/fitted['scale'])
        control = fitted['control'](x).log_softmax(-1).numpy()
        logits = np.concatenate([h(x).numpy() for h in fitted['binaries']],axis=1)
    return control, coupled_logits(control,logits,fitted['pairs'])


def serialize(h):
    return dict(hidden=h[0].weight.detach().tolist(),hiddenBias=h[0].bias.detach().tolist(),
        output=h[2].weight.detach().tolist(),outputBias=h[2].bias.detach().tolist() if h[2].bias is not None else [0.],
        outputHasBias=h[2].bias is not None)


def train(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    data = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    if len({r.get('id') for r in data}) != len(data):
        raise ValueError('Duplicate recording identity')
    rows, skipped = [], []
    for r in data:
        if r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded or not r.get('id'):
            raise ValueError('Ineligible training provenance')
        if r['kind'] not in ('none','substituted_word'):
            raise ValueError('Unsupported script kind')
        x = np.asarray(r['features'])
        if x.ndim != 2 or x.shape[1] != DIMENSIONS or not np.isfinite(x).all():
            raise ValueError('Invalid frozen features')
        if r['kind'] == 'none':
            found = occupancies(r['logProbs'],r['pronunciation'])
            if found is None:
                skipped.append(r['id']);continue
            targets,_ = found
            if len(targets) != len(x) or targets.shape[1] != len(metadata['tokens'])+1:
                raise ValueError('Native feature/occupancy mismatch')
            rows.append(dict(r,targets=targets))
    anchors = sorted({r['speaker'] for r in rows})
    if len(anchors) < 3:
        raise ValueError('Three none-labelled training speakers required')
    speakers = anchors + sorted(set(r['speaker'] for r in data)-set(anchors))
    folds = {s:i%3 for i,s in enumerate(speakers)}
    controls = [dict(c,speaker=r['speaker']) for r in data if r['kind']=='substituted_word' for c in r['controls']]
    validation = []
    for reg in REGULARIZATION:
        errors, control_errors, protected, capacities = {}, {}, [], []
        for fold in range(3):
            fitted = fit([r for r in rows if folds[r['speaker']] != fold],metadata['tokens'],reg)
            capacities.append(fitted['parameters'])
            for r in rows:
                if folds[r['speaker']] == fold:
                    a,b = outputs(fitted,r['features'])
                    errors.setdefault(r['speaker'],[]).append(float(-(r['targets']*b).sum(1).mean()))
                    control_errors.setdefault(r['speaker'],[]).append(float(-(r['targets']*a).sum(1).mean()))
            for c in controls:
                if folds[c['speaker']] == fold:
                    a,b = outputs(fitted,c['features'])
                    margins = [sequence_score(lp,c['written'])-sequence_score(lp,c['rival']) for lp in (a,b)]
                    if all(np.isfinite(margins)):
                        protected.append(margins[0]<0 and margins[1]>0)
        item = dict(regularization=reg,meanSpeakerSoftNll=float(np.mean([np.mean(v) for v in errors.values()])),
            controlMeanSpeakerSoftNll=float(np.mean([np.mean(v) for v in control_errors.values()])),
            preservationComparisons=len(protected),preservationReversals=sum(protected),foldParameters=capacities)
        validation.append(item);print(json.dumps(item),flush=True)
    if not any(v['preservationComparisons'] for v in validation):
        raise ValueError('No real substitution preservation comparisons available')
    selected = min(validation,key=lambda v:(v['preservationReversals']/v['preservationComparisons'],v['meanSpeakerSoftNll'],v['regularization']))
    fitted = fit(rows,metadata['tokens'],selected['regularization'])
    metadata.update(center=fitted['center'].tolist(),scale=fitted['scale'].tolist(),control=serialize(fitted['control']),
        pairs=[dict(a=a,b=b,head=serialize(h)) for (a,b),h in zip(fitted['pairs'],fitted['binaries'])],parameters=fitted['parameters'])
    out = Path(args.out)
    asset = out/'pairwise-phone-heads.json'
    asset.write_text(json.dumps(metadata,allow_nan=False))
    report = dict(recordings=len(rows),speakers=anchors,excludedSpeakers=sorted(excluded),folds=folds,impossibleAlignments=skipped,
        controls=len(controls),validation=validation,selected=selected,pairs=fitted['pairs'],parametersPerArm=fitted['parameters'],epochs=EPOCHS,
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'pairwise-phone-training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    files = [asset,out/'pairwise-phone.onnx']
    if not all(p.is_file() for p in files):
        raise ValueError('Missing frozen feature graph')
    (out/'pairwise-phone-asset.json').write_text(json.dumps(dict(schema=1,files=[dict(file=p.name,bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])))
    print(json.dumps(report),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('input','metadata','excluded-speakers','out'):
        p.add_argument('--'+name,required=True)
    train(p.parse_args())
