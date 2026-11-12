"""Real-word audio negatives on a frozen encoder; no practice labels enter fitting.

Pairing, speaker partitions, architecture and checkpoint selection are fixed here.
Both arms use precisely the same complete recorded-word positives. Matched swaps
are reciprocal; random swaps are reciprocal on the same retained recording set.
"""
import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

SEED = 1729
EPOCHS = 24
VOWELS = {'َ': 'a', 'ُ': 'u', 'ِ': 'i', 'ا': 'a', 'ۦ': 'i', 'ۥ': 'u', '۪': 'e', 'ـ': 'e', 'ٲ': 'a'}
CONSONANTS = set('ءبتثجحخدذرزسشصضطظعغفقكلمنهوي')


def vowel_sequence(ids, tokens):
    # Collapse held vowel runs; consonantal waw/ya are not asserted to be vowels.
    result, previous = [], None
    for i in ids:
        p = VOWELS.get(tokens[i-1])
        if p and p != previous: result.append(p)
        previous = p
    return tuple(result)


def partition(speaker):
    return int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) % 5


def consonant_sequence(ids, tokens):
    result, previous = [], None
    for i in ids:
        p = tokens[i-1]
        if p in CONSONANTS and p != previous: result.append(p)
        previous = p if p in CONSONANTS else None
    return tuple(result)


def match_words(words, tokens):
    """Disjoint actual-audio swaps, within speaker/capture and vowel signature.

Duration tolerance and descriptor bound are preregistered, never target-selected.
Same-take pairs have exactly the same recording characteristics. Otherwise require
RMS within a factor e and mean frozen-feature descriptor distance <= 3.
"""
    remaining = set(range(len(words)))
    pairs = []
    for a in range(len(words)):
        if a not in remaining: continue
        x = words[a]
        eligible = []
        for b in sorted(remaining):
            if b == a: continue
            y = words[b]
            if len(x['x']) < len(y['ids'])+1 or len(y['x']) < len(x['ids'])+1: continue
            if x['speaker'] != y['speaker'] or x['capture'] != y['capture'] or x['ids'] == y['ids']: continue
            if consonant_sequence(x['ids'], tokens) == consonant_sequence(y['ids'], tokens): continue
            if vowel_sequence(x['ids'], tokens) != vowel_sequence(y['ids'], tokens): continue
            ratio = max(len(x['x']), len(y['x'])) / min(len(x['x']), len(y['x']))
            if ratio > 1.3: continue
            delta = math.sqrt(sum((u-v)**2 for u,v in zip(x['descriptor'][1:],y['descriptor'][1:])) / 8)
            if abs(x['descriptor'][0]-y['descriptor'][0]) > 1 or delta > 3: continue
            cost = abs(math.log(ratio)) + .1*delta + (0 if x['take'] == y['take'] else .1)
            eligible.append((cost, b))
        if eligible:
            b = min(eligible)[1]
            pairs.append((a,b)); remaining.remove(a); remaining.remove(b)
    return pairs


def random_pairs(words, matched):
    """Random reciprocal swaps on the identical positive recording multiset.

Fifty sweeps of symmetric pair-switch proposals sample valid perfect matchings;
only different pronunciation and enough frames constrain the control. No speaker,
capture, vowel signature or duration similarity is consulted by this arm.
"""
    swaps = list(matched)
    rng = random.Random(SEED)
    if len(swaps)<2: raise ValueError('Need two pairs for a random-negative control')
    def valid(a,b):
        return words[a]['ids'] != words[b]['ids'] and len(words[a]['x']) >= len(words[b]['ids'])+1 and len(words[b]['x']) >= len(words[a]['ids'])+1
    for _ in range(50*len(swaps)):
        i,j = rng.sample(range(len(swaps)),2)
        a,b = swaps[i]; c,d = swaps[j]
        if rng.randrange(2): c,d = d,c
        if valid(a,c) and valid(b,d): swaps[i],swaps[j]=(a,c),(b,d)
    if sorted(i for pair in swaps for i in pair) != sorted(i for pair in matched for i in pair):
        raise ValueError('Random arm changed positive multiset')
    return swaps


def make_head(vocabulary):
    import torch
    from torch import nn
    class Compatibility(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary,16)
            self.hidden = nn.Linear(48,32)
            self.output = nn.Linear(32,1)
        def forward(self,x,ids):
            n,p = x.shape[0],ids.shape[0]
            query = self.embedding(ids)[:,None,:].expand(p,n,16)
            acoustic = x[None,:,:].expand(p,n,32)
            e = self.output(self.hidden(torch.cat((acoustic,query),-1)).tanh()).squeeze(-1)/n
            # All phones consume >=1 frame; forward sum integrates every boundary.
            # Finite unreachable sentinel avoids NaN derivatives of (-inf,-inf).
            dp = torch.cat((e[:1,0],torch.full((p-1,),-1e4,device=x.device)))
            for t in range(1,n):
                advance = torch.cat((torch.full((1,),-1e4,device=x.device),dp[:-1]))
                dp = torch.logaddexp(dp,advance) + e[:,t]
            prior = math.lgamma(n)-math.lgamma(p)-math.lgamma(n-p+1)
            return dp[-1]-prior
    return Compatibility()


def predict(head,x,ids):
    import torch
    n,p = len(x),len(ids)
    views = [(0,n),(1,n),(0,n-1)]
    logits = [head(x[a:b],ids) for a,b in views if b-a >= p]
    return torch.logsumexp(torch.stack(logits),0)-math.log(len(logits))


def serialize(head):
    return dict(embedding=head.embedding.weight.detach().tolist(), weight=head.hidden.weight.detach().tolist(),
      bias=head.hidden.bias.detach().tolist(), output=head.output.weight.detach()[0].tolist(),
      intercept=float(head.output.bias.detach()[0]))


def fit(args):
    import numpy as np
    import torch
    torch.set_num_threads(1); torch.manual_seed(SEED)
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s]
    for r in rows:
        if r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or r.get('speaker') in excluded or not r.get('speaker') or r.get('kind') != 'none':
            raise ValueError('Invalid positive training provenance')
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < 6: raise ValueError('Insufficient independent training speakers')
    fit_rows = [r for r in rows if partition(r['speaker']) < 3]
    if not fit_rows: raise ValueError('Empty fitting partition')
    features = np.concatenate([r['features'] for r in fit_rows]).astype(np.float32)
    center,scale = features.mean(0),np.maximum(features.std(0),.01)
    metadata.update(center=center.tolist(),scale=scale.tolist())
    partitions = {}
    audit = {}
    for group in ('fit','select','audit'):
        chosen = [r for r in rows if ('fit' if partition(r['speaker'])<3 else 'select' if partition(r['speaker'])==3 else 'audit') == group]
        words = []
        for r in chosen:
            for w in r['words']:
                x = np.asarray(r['features'][w['from']:w['to']],dtype=np.float32)
                ids = w['ids']
                if len(ids)<2 or len(x)<len(ids)+1 or len(x)>100: continue
                if any(i<1 or i>len(metadata['tokens']) for i in ids): raise ValueError('Unknown positive phone')
                words.append(dict(x=x,ids=ids,speaker=r['speaker'],capture=r['capture'],descriptor=r['descriptor'],take=r['id']))
        # Bound the cached head budget independently of targets.
        words = words[:4000]
        pairs = match_words(words,metadata['tokens'])
        random_arm = random_pairs(words,pairs)
        if not pairs: raise ValueError('No actual duration/vowel/capture-matched word pairs in '+group)
        examples = {}
        for arm,swaps in [('matched',pairs),('random',random_arm)]:
            data = []
            for a,b in swaps:
                for positive,negative in [(a,b),(b,a)]:
                    ids = torch.tensor(words[positive]['ids'],dtype=torch.int64)
                    for audio,label in [(positive,1),(negative,0)]:
                        x = torch.tensor((words[audio]['x']-center)/scale)
                        if len(x)<len(ids): raise ValueError('Negative cannot consume complete candidate')
                        data.append((x,ids,float(label)))
            examples[arm]=data
        if len(examples['matched'])!=len(examples['random']): raise ValueError('Unequal arm budget')
        partitions[group] = examples
        audit[group] = dict(speakers=sorted({r['speaker'] for r in chosen}),words=len(words),pairs=len(pairs),
          examples=len(examples['matched']),retainedWords=len(pairs)*2,
          matchedPairs=[dict(first=words[a]['take'],second=words[b]['take'],candidate=words[a]['ids'],rival=words[b]['ids'],
            firstFrames=len(words[a]['x']),secondFrames=len(words[b]['x'])) for a,b in pairs])
    initial = make_head(len(metadata['tokens'])+1).state_dict()
    out = Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report = dict(seed=SEED,epochs=EPOCHS,partitions=audit,excludedSpeakers=sorted(excluded),
      inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),arms={})
    for arm in ('matched','random'):
        head = make_head(len(metadata['tokens'])+1);head.load_state_dict(initial)
        opt = torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.01)
        data = partitions['fit'][arm]; rng=random.Random(SEED)
        best,loss_best,history = None,float('inf'),[]
        for epoch in range(EPOCHS):
            order = list(range(len(data)));rng.shuffle(order);head.train()
            for offset in range(0,len(order),16):
                opt.zero_grad();batch=order[offset:offset+16]
                losses = [torch.nn.functional.binary_cross_entropy_with_logits(predict(head,data[i][0],data[i][1]),torch.tensor(data[i][2])) for i in batch]
                loss = torch.stack(losses).mean();loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),5);opt.step()
            head.eval()
            with torch.no_grad():
                loss = float(torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(predict(head,x,ids),torch.tensor(y)) for x,ids,y in partitions['select'][arm]]).mean())
            history.append(dict(epoch=epoch+1,selectionBCE=loss))
            if loss<loss_best: best=copy.deepcopy(head.state_dict());loss_best=loss
            print(json.dumps(dict(arm=arm,**history[-1])),flush=True)
        head.load_state_dict(best);head.eval()
        with torch.no_grad():
            observed = [dict(score=float(predict(head,x,ids)),label=y) for x,ids,y in partitions['audit'][arm]]
        (out/('duration-negatives-'+arm+'.json')).write_text(json.dumps(serialize(head),allow_nan=False))
        report['arms'][arm]=dict(history=history,selectedEpoch=min(history,key=lambda r:r['selectionBCE'])['epoch'],audit=observed)
        # Export-parity fixtures are training-only, never used to adjust selection.
        fixture = []
        with torch.no_grad():
            for x,ids,y in partitions['audit'][arm][:12]:
                fixture.append(dict(features=(x.numpy()*scale+center).tolist(),ids=ids.tolist(),score=float(predict(head,x,ids))))
        (out/('duration-negatives-'+arm+'-parity.json')).write_text(json.dumps(fixture))
    (out/'duration-negatives.json').write_text(json.dumps(metadata,allow_nan=False))
    names=['duration-negatives.json','duration-negatives-features.onnx','duration-negatives-matched.json','duration-negatives-random.json']
    assets={n:dict(bytes=(out/n).stat().st_size,sha256=hashlib.sha256((out/n).read_bytes()).hexdigest()) for n in names}
    (out/'duration-negatives-assets.json').write_text(json.dumps(assets))
    (out/'duration-negatives-training-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(speakers=speakers,retained={k:v['retainedWords'] for k,v in audit.items()})),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('input','metadata','excluded-speakers','out'):p.add_argument('--'+name,required=True)
    fit(p.parse_args())
