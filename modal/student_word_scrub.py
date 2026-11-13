"""Conditional lexical removal on cached frozen features; no practice inputs.

Phone supervision is a complete transcript CTC chain plus its soft occupancies.
Only multiply-attested phones/words/speakers enter adversarial supervision.
All arms have the same residual projection, phone head and update schedule.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path


def stable(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def write_json(path, value):
    target = Path(path)
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temp.replace(target)


def partition(rows, excluded):
    for r in rows:
        if (r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('kind') != 'none' or
                not r.get('speaker') or r['speaker'] in excluded or r.get('excluded') or
                r.get('events') or r.get('non_errors') or r.get('unsupported')):
            raise ValueError('Unauthorized training provenance')
    speakers = sorted({r['speaker'] for r in rows}, key=stable)
    if len(speakers) < 4 or len(rows) < 8 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Need unique recordings from at least four training speakers')
    valid = set(speakers[:max(1, len(speakers)//5)])
    return [r for r in rows if r['speaker'] not in valid], [r for r in rows if r['speaker'] in valid]


def eligible_labels(rows):
    """A phone must span words and speakers; each phone-word pair is independently
    attested by multiple speakers. Count transcript occurrences, never frames."""
    groups = {}
    for r in rows:
        for p, word in zip(r['ids'], r['wordLabels']):
            if word is not None:
                groups.setdefault(p, {}).setdefault(word, set()).add(r['speaker'])
    return {p: sorted(w for w, speakers in words.items() if len(speakers) >= 2)
            for p, words in groups.items() if sum(len(s) >= 2 for s in words.values()) >= 2}


def shuffle_words(records):
    """Destroy lexical/acoustic pairing conditional on both phone and speaker.
    The phone, speaker and word marginals are exactly retained."""
    import numpy as np
    result = [dict(r) for r in records]
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault((r['phone'], r['speaker']), []).append(i)
    for key, indices in sorted(groups.items()):
        rng = np.random.default_rng(int(stable(key)[:8], 16))
        values = [records[i]['word'] for i in indices]
        for i, j in zip(indices, rng.permutation(len(indices))):
            result[i]['word'] = values[j]
    return result


def occupancies(lp, ids):
    """Exact forward/backward over a complete CTC transcript, including blanks.
    Returns soft state placements, not engine word boundaries or argmax spikes."""
    import numpy as np
    lp = np.asarray(lp, dtype=np.float64)
    labels = np.array([0, *[v for p in ids for v in (p, 0)]])
    n, s = len(lp), len(labels)
    skip = (labels != 0) & (labels != np.roll(labels, 2)); skip[:2] = False
    forward = np.full((n, s), -np.inf)
    forward[0, :2] = lp[0, labels[:2]]
    for t in range(1, n):
        previous = forward[t-1]
        a = np.logaddexp(previous, np.r_[-np.inf, previous[:-1]])
        forward[t] = np.logaddexp(a, np.where(skip, np.r_[[-np.inf]*2, previous[:-2]], -np.inf)) + lp[t, labels]
    z = np.logaddexp(forward[-1, -1], forward[-1, -2])
    if not np.isfinite(z):
        raise ValueError('Impossible complete transcript')
    backward = np.full((n, s), -np.inf); backward[-1, -2:] = 0
    for t in range(n-2, -1, -1):
        following = backward[t+1] + lp[t+1, labels]
        a = np.logaddexp(following, np.r_[following[1:], -np.inf])
        backward[t] = np.logaddexp(a, np.r_[np.where(skip[2:], following[2:], -np.inf), [-np.inf]*2])
    return np.exp(forward + backward - z), float(z)


def fit(args):
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.process_time()
    metadata = json.loads(Path(args.metadata).read_text())
    extraction = json.loads(Path(args.extraction_report).read_text())
    spent = metadata['exportCpuSeconds'] + extraction['cpuSeconds']
    def budget():
        if spent + time.process_time() - start >= 86400:
            raise TimeoutError('One cached-feature CPU day exhausted')
    rows = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    train, valid = partition(rows, excluded)
    controls = [json.loads(s) for s in Path(args.preservation).read_text().splitlines() if s.strip()]
    for r in controls:
        if r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('kind') != 'scripted-control' or r['speaker'] in excluded:
            raise ValueError('Unauthorized substitution preservation control')
    valid_speakers = {r['speaker'] for r in valid}
    controls = [r for r in controls if r['speaker'] in valid_speakers]
    if not controls: raise ValueError('No held-out training-speaker scripted substitution controls')
    vocab = len(metadata['tokens']) + 1
    for r in rows:
        x, lp = np.asarray(r['features']), np.asarray(r['logProbs'])
        if (x.ndim != 2 or x.shape[1] != 32 or lp.shape != (len(x), vocab) or
                not np.isfinite(x).all() or not np.isfinite(lp).all() or not r['ids'] or
                len(r['ids']) != len(r['wordLabels']) or any(i < 1 or i >= vocab for i in r['ids'])):
            raise ValueError('Malformed cached features/transcript')
    joined = np.concatenate([r['features'] for r in train])
    center, scale = joined.mean(0), np.maximum(joined.std(0), .01)
    eligible = eligible_labels(train)
    words = sorted({w for values in eligible.values() for w in values})
    if len(words) < 2:
        raise ValueError('No multiword, multispeaker conditional supervision')
    word_index = {w: i for i, w in enumerate(words)}
    def prepare(population):
        data = []
        for r in population:
            budget()
            q, _ = occupancies(r['logProbs'], r['ids'])
            target = np.zeros((len(q), vocab))
            labels = [0, *[v for p in r['ids'] for v in (p, 0)]]
            for s, p in enumerate(labels): target[:, p] += q[:, s]
            positions = [i for i, (p, w) in enumerate(zip(r['ids'], r['wordLabels'])) if w in eligible.get(p, [])]
            pooling = q[:, [2*i+1 for i in positions]].T
            pooling /= np.maximum(pooling.sum(1, keepdims=True), 1e-12)
            data.append(dict(x=torch.tensor((np.asarray(r['features'])-center)/scale, dtype=torch.float32),
                y=torch.tensor(r['ids']), target=torch.tensor(target, dtype=torch.float32),
                pooling=torch.tensor(pooling, dtype=torch.float32),
                phone=torch.tensor([r['ids'][i] for i in positions], dtype=torch.long),
                word=torch.tensor([word_index[r['wordLabels'][i]] for i in positions], dtype=torch.long), speaker=r['speaker']))
        return data
    training, validation = prepare(train), prepare(valid)
    occurrences = [dict(phone=int(p), speaker=r['speaker'], word=int(w)) for r in training for p, w in zip(r['phone'], r['word'])]
    shuffled = shuffle_words(occurrences); at = 0
    for r in training:
        n = len(r['word']); r['shuffled'] = torch.tensor([v['word'] for v in shuffled[at:at+n]], dtype=torch.long); at += n
    # Balance lexical outcomes conditional on phone; no frequent word can dominate.
    counts = {}
    for r in occurrences: counts[(r['phone'], r['word'])] = counts.get((r['phone'], r['word']), 0) + 1
    def weights(phone, word):
        w = torch.tensor([1/counts.get((int(p), int(v)), 1) for p, v in zip(phone, word)])
        return w / w.mean()
    class Readout(nn.Module):
        def __init__(self):
            super().__init__(); self.hidden = nn.Linear(32, 32); self.residual = nn.Linear(32, 32, bias=False)
            nn.init.zeros_(self.residual.weight); self.output = nn.Linear(32, vocab)
        def project(self, x): return x + self.residual(self.hidden(x).tanh())
        def forward(self, x): return self.output(self.project(x)).log_softmax(-1)
    class Adversary(nn.Module):
        def __init__(self, feature=True):
            super().__init__(); self.feature = feature
            self.head = nn.Sequential(nn.Linear(vocab + (32 if feature else 0), 64), nn.Tanh(), nn.Linear(64, len(words)))
        def forward(self, z, phone):
            identity = torch.nn.functional.one_hot(phone, vocab).float()
            return self.head(torch.cat([z, identity], -1) if self.feature else identity)
    ctc = nn.CTCLoss(blank=0, reduction='sum')
    def phone_loss(model, r):
        lp = model(r['x'])
        return ctc(lp[:, None], r['y'], [len(lp)], [len(r['y'])])/len(lp) - (lp*r['target']).sum()/len(lp)
    def lexical_loss(adv, z, r, labels):
        return (torch.nn.functional.cross_entropy(adv(z, r['phone']), labels, reduction='none') * weights(r['phone'], labels)).mean()
    arms = ('residual', 'original', 'shuffled'); models, advs, optimizers, adv_optimizers = {}, {}, {}, {}
    for arm in arms:
        torch.manual_seed(1729); models[arm] = Readout(); advs[arm] = Adversary()
        optimizers[arm] = torch.optim.AdamW(models[arm].parameters(), lr=.001, weight_decay=.01)
        adv_optimizers[arm] = torch.optim.AdamW(advs[arm].parameters(), lr=.001, weight_decay=.01)
    def control_vetoes():
        won, total = 0, 0
        with torch.no_grad():
            for r in controls:
                budget()
                features = np.asarray(r['features'])
                if features.ndim != 2 or features.shape[1] != 32 or not np.isfinite(features).all():
                    raise ValueError('Invalid preservation acoustics')
                x = torch.tensor((features-center)/scale, dtype=torch.float32)
                lp = {a: models[a](x).numpy() for a in arms}
                for c in r['candidates']:
                    bounds = c['fromFrame'], c['toFrame']
                    if not 0 <= bounds[0] < bounds[1] <= len(x): continue
                    labels = [[metadata['tokens'].find(p)+1 for p in c[k]] for k in ('written', 'rival')]
                    if any(not y or any(p < 1 for p in y) for y in labels): continue
                    margins = []
                    try:
                        for arm in arms:
                            scores = [occupancies(lp[arm][bounds[0]:bounds[1]], y)[1] for y in labels]
                            margins.append(scores[0]-scores[1])
                    except ValueError: continue
                    total += 1; won += margins[0] > 0 and margins[1] < 0 and margins[2] < 0
        return dict(vetoed=int(won), comparisons=total)
    rng = np.random.default_rng(1729); choices, best = [], None
    for epoch in range(80):
        for i in rng.permutation(len(training)):
            budget(); r = training[i]
            for arm in arms:
                model, adv = models[arm], advs[arm]
                labels = r['shuffled'] if arm == 'shuffled' else r['word']
                if len(labels):
                    adv_optimizers[arm].zero_grad(set_to_none=True)
                    lexical_loss(adv, r['pooling'] @ model.project(r['x']).detach(), r, labels).backward()
                    adv_optimizers[arm].step()
                optimizers[arm].zero_grad(set_to_none=True)
                value = phone_loss(model, r)
                if len(labels) and arm != 'original':
                    for p in adv.parameters(): p.requires_grad_(False)
                    value = value - .1 * lexical_loss(adv, r['pooling'] @ model.project(r['x']), r, labels)/np.log(len(words))
                if not torch.isfinite(value): raise ValueError('Nonfinite training loss')
                value.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); optimizers[arm].step()
                for p in adv.parameters(): p.requires_grad_(True)
        if (epoch+1) % 10 == 0:
            with torch.no_grad(): scores = {a: float(torch.stack([phone_loss(models[a], r) for r in validation]).mean()) for a in arms}
            control = control_vetoes()
            joint = sum(scores.values())/3 + .1 * control['vetoed']/max(1, control['comparisons'])
            choice = dict(epoch=epoch+1, validation=scores, substitutionPreservation=control, joint=joint); choices.append(choice)
            if best is None or joint < best[0]: best = joint, choice, {a: copy.deepcopy(models[a].state_dict()) for a in arms}
            print(json.dumps(choice), flush=True)
    for a in arms: models[a].load_state_dict(best[2][a])
    # Fresh probes are fitted after projection selection, on fitting speakers only.
    # They cannot change the checkpoint. A phone-only probe reports unavoidable
    # lexical predictability; the acoustic probe tests information beyond identity.
    def probe_data(data, model):
        with torch.no_grad(): z = torch.cat([r['pooling'] @ model.project(r['x']) for r in data])
        return z, torch.cat([r['phone'] for r in data]), torch.cat([r['word'] for r in data])
    probes = {}
    for name in (*arms, 'phoneOnly'):
        model = models['original' if name == 'phoneOnly' else name]
        x, p, y = probe_data(training, model); vx, vp, vy = probe_data(validation, model)
        if not len(vy): raise ValueError('No cross-speaker validation lexical overlap')
        torch.manual_seed(8191); probe = Adversary(name != 'phoneOnly'); optimizer = torch.optim.AdamW(probe.parameters(), lr=.003, weight_decay=.01)
        for _ in range(120):
            budget(); optimizer.zero_grad(set_to_none=True)
            loss = (torch.nn.functional.cross_entropy(probe(x, p), y, reduction='none')*weights(p, y)).mean()
            loss.backward(); optimizer.step()
        with torch.no_grad():
            lp = probe(vx, vp).log_softmax(-1)
            probes[name] = dict(nll=float((-lp[torch.arange(len(vy)), vy]*weights(vp, vy)).mean()),
                accuracy=float((lp.argmax(-1)==vy).float().mean()), validationOccurrences=len(vy))
    # Training-only counterfactual preservation: replace each supervised consonant
    # with another observed consonant in the same validation transcript. These are
    # weak acoustic controls, not assertions about performed scripted mistakes.
    consonants = [i+1 for i, p in enumerate(metadata['tokens']) if p in 'ءبتثجحخدذرزسشصضطظعغفقكلمنهوي']
    preservation = {}
    for arm in arms:
        wins, total = 0, 0
        for r in validation:
            budget()
            lp = models[arm](r['x']).detach().numpy()
            _, written = occupancies(lp, r['y'].tolist())
            for i, p in enumerate(r['y'].tolist()):
                if p not in consonants: continue
                rival = r['y'].tolist(); rival[i] = consonants[(consonants.index(p)+1)%len(consonants)]
                try: _, score = occupancies(lp, rival)
                except ValueError: continue
                total += 1; wins += written > score
        preservation[arm] = dict(writtenPreferred=wins, comparisons=total)
    budget()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    result = {k: metadata[k] for k in ('schema', 'inventory', 'tokens', 'primarySha256', 'featureOutput', 'projection')}
    result.update(center=center.tolist(), scale=scale.tolist()); fixtures = []
    for arm in arms:
        sd = models[arm].state_dict()
        result[arm] = dict(weight=sd['hidden.weight'].tolist(), bias=sd['hidden.bias'].tolist(), residual=sd['residual.weight'].tolist(),
            output=sd['output.weight'].tolist(), intercept=sd['output.bias'].tolist())
        for r, cached in zip(valid[:3], validation[:3]):
            _, score = occupancies(models[arm](cached['x']).detach().numpy(), r['ids'])
            fixtures.append(dict(arm=arm, features=r['features'], ids=r['ids'], score=score))
    write_json(out/'word-scrub.json', result); write_json(out/'word-scrub-parity.json', fixtures)
    assets = {}
    for name in ('word-scrub.json', 'word-scrub-features.onnx'):
        data = (out/name).read_bytes(); assets[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    write_json(out/'word-scrub-assets.json', assets)
    report = dict(selected=best[1], choices=choices, selection='joint held-out training-speaker phone loss plus weak substitution retention',
        trainingSpeakers=sorted({r['speaker'] for r in train}), validationSpeakers=sorted({r['speaker'] for r in valid}),
        trainingRecordings=len(train), validationRecordings=len(valid), eligiblePhones=eligible, conditionalProbes=probes,
        reducedConditionalPredictability=(probes['original']['nll'] < probes['phoneOnly']['nll'] and
            probes['residual']['nll'] > probes['original']['nll'] and probes['residual']['nll'] > probes['shuffled']['nll']),
        counterfactualConsonantControls=preservation, weakScriptedSubstitutionPreservation=control_vetoes(), parametersPerReadout=sum(p.numel() for p in Readout().parameters()),
        adversaryWeight=.1, cpuSeconds=spent+time.process_time()-start, inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    write_json(out/'word-scrub-training-report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('input', 'metadata', 'extraction-report', 'excluded-speakers', 'preservation', 'out'): p.add_argument('--'+name, required=True)
    fit(p.parse_args())
