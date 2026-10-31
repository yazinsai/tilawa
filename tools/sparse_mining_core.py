"""Occurrence-aware sparse mining. Pure helpers; never infer a transcript from canonical text."""
import collections
import hashlib
import random


def runs(ids):
    """Keep start/end frames, including blank boundaries between repeated symbols."""
    out=[]
    for frame, token in enumerate(ids):
        if out and out[-1]['token']==token and out[-1]['end']==frame:
            out[-1]['end']=frame+1
        else:
            out.append(dict(token=int(token),start=frame,end=frame+1))
    return [r for r in out if r['token']]


def word_target(words, encode):
    """Collapse duration spelling inside each word, never across word boundaries."""
    target=[]
    for wi,w in enumerate(words):
        for token, group in __import__('itertools').groupby(encode(w['phones'])):
            target.append(dict(token=token,word=wi,referenceDuration=len(list(group))))
    return target


def duration_groups(emissions):
    """Retain repeated acoustic emissions as duration evidence for one token.

    This never merges source word records. A source boundary with the same
    token on both sides remains ambiguous and is rejected by attach_timing.
    """
    out=[]
    for r in emissions:
        if out and out[-1]['token']==r['token']:
            out[-1]['end']=r['end'];out[-1]['emissions'].append(r)
        else:out.append({**r,'emissions':[r]})
    return out


def attach_timing(target, clean_runs):
    grouped=duration_groups(clean_runs)
    if [x['token'] for x in target] != [x['token'] for x in grouped]:
        return None
    return [{**a,**b,'occurrence':i,'short':b['end']-b['start']<=2} for i,(a,b) in enumerate(zip(target,grouped))]


def failures(timed, degraded, margin_frames=10):
    """DP matches only same-token occurrences within 400 ms of their clean position.

    It consumes every match once. Extra consonants and absent/changed consonants
    retain occurrence indices; repeated symbols cannot reuse one shared emission.
    """
    n,m=len(timed),len(degraded)
    costs=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):costs[i][0]=i
    for j in range(m+1):costs[0][j]=j
    def near(a,b): return a['start']-margin_frames<=b['start']<=a['end']+margin_frames
    for i in range(1,n+1):
        for j in range(1,m+1):
            match=0 if timed[i-1]['token']==degraded[j-1]['token'] and near(timed[i-1],degraded[j-1]) else 3
            costs[i][j]=min(costs[i-1][j]+1,costs[i][j-1]+1,costs[i-1][j-1]+match)
    i,j=n,m; missing=[];extra=[];matches=[]
    while i or j:
        if i and j and timed[i-1]['token']==degraded[j-1]['token'] and near(timed[i-1],degraded[j-1]) and costs[i][j]==costs[i-1][j-1]:
            matches.append((i-1,j-1));i-=1;j-=1
        elif i and costs[i][j]==costs[i-1][j]+1:missing.append(i-1);i-=1
        else:extra.append(j-1);j-=1
    return {'missing':sorted(missing),'extra':sorted(extra),'matches':list(reversed(matches))}


def persistent(timed, views, consonants):
    counts=collections.Counter(i for v in views for i in v['missing'] if timed[i]['token'] in consonants)
    return [i for i,n in sorted(counts.items()) if n>=2]


def freeze_arms(keys, candidates, cap=256, fraction=.02, seed=2718):
    n=min(cap,int(len(keys)*fraction),len(candidates))
    if n<1: raise ValueError('No eligible persistent failures')
    rng=random.Random(seed)
    # Random choice among qualifying failures avoids unbounded hardest-case selection.
    selected=sorted(rng.sample(sorted(candidates),n))
    # Independent random subset of full eligible pool; overlap is valid and recorded.
    random_keys=sorted(random.Random(seed+1).sample(sorted(keys),n))
    return {'selected':selected,'random':random_keys,'n':n,'weight':3.,'seed':seed,
            'overlap':sorted(set(selected)&set(random_keys)),
            'poolSha256':hashlib.sha256('\n'.join(keys).encode()).hexdigest()}


def validated_weights(keys, manifest, arm):
    """Fail closed before any update if provenance, targets, or controls are missing."""
    if arm not in ('selected','random'): raise ValueError('Unknown sparse arm')
    digest=hashlib.sha256('\n'.join(keys).encode()).hexdigest()
    if manifest.get('poolSha256') != digest: raise ValueError('Training pool changed')
    policy=manifest['policy']
    if policy != {'ctcWeight':1.,'kdWeight':2.,'updates':200,'weight':3.}:
        raise ValueError('Original objective and budget must remain fixed')
    if len(manifest['selected']) != len(manifest['random']): raise ValueError('Unmatched arm sizes')
    if len(manifest['selected']) > min(128,int(.02*len(keys))): raise ValueError('Sparse cap exceeded')
    if not manifest['selected']:raise ValueError('Empty selected set')
    for group in ('selected','random'):
        if len(set(manifest[group]))!=len(manifest[group]):raise ValueError('Duplicate key')
        for key in manifest[group]:
            if key not in keys:raise ValueError('Out-of-pool key')
            proof=manifest.get('verifiedTargets',{}).get(key,{})
            if not proof.get('audioSha256') or not proof.get('independentAsSaidMatch') or not proof.get('reviewRecord'):
                raise ValueError('Unverified as-said target: '+key)
    controls=manifest.get('selfCorrectionControls',[])
    if len(controls)<2 or not all(c.get('verified') and c.get('audioSha256') and c.get('reviewRecord') for c in controls):
        raise ValueError('Two verified self-correction controls are required')
    chosen=set(manifest[arm])
    return [3. if key in chosen else 1. for key in keys]
