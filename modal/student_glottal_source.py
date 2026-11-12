"""Paired source-filter views and frozen-encoder CTC readouts.

Only usable, none-labelled help-train training speakers enter this utility.
Both arms see byte-identical cached examples, initial weights and update order.
The sole arm difference is occurrence-pooled identity consistency. No practice
audio, labels, warning locations or text alternatives enter model selection.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def eligible(row, excluded):
    return (row.get('kind') == 'none' and row.get('split') == 'train'
            and bool(row.get('speaker')) and row['speaker'] not in excluded
            and not row.get('exclude') and not row.get('excluded')
            and row.get('scored') is not False and not row.get('events')
            and not row.get('non_errors'))


def speaker_folds(speakers):
    speakers = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(speakers) < 6:
        raise ValueError('Need at least six eligible training speakers')
    return {s: i % 3 for i, s in enumerate(speakers)}


def lpc(frame, order=20):
    """Regularized autocorrelation LPC with stable reflection coefficients."""
    import numpy as np
    x = (frame-frame.mean()) * np.hanning(len(frame))
    r = np.array([np.dot(x[k:],x[:len(x)-k]) for k in range(order+1)])
    r[0] += max(r[0]*1e-4, 1e-10)
    a, error = np.array([1.]), r[0]
    for k in range(1, order+1):
        reflection = np.clip(-(r[k]+np.dot(a[1:], r[k-1:0:-1]))/error, -.98, .98)
        a = np.r_[a, 0.] + reflection*np.r_[0., a[::-1]]
        error *= 1-reflection**2
    return a


def excitation_analysis(audio, rate=16000):
    """Text-free conservative voicing and pitch. Protect low-confidence boundaries."""
    import numpy as np
    from scipy.signal import butter, sosfilt, find_peaks, correlate
    x = sosfilt(butter(4, 900, fs=rate, output='sos'), audio)
    hop, half = rate//100, rate//25
    centers = np.arange(0, len(audio), hop)
    pitch, voiced = np.zeros(len(centers)), np.zeros(len(centers), dtype=bool)
    lo, hi = rate//400, rate//65
    for j, c in enumerate(centers):
        f = x[max(0,c-half):min(len(x),c+half)]
        if len(f) < 2*hi or np.sqrt(np.mean(f*f)) < .003:
            continue
        f = f-f.mean()
        corr = correlate(f,f,'full',method='fft')[len(f)-1:]
        corr /= max(corr[0], 1e-12)
        peaks, _ = find_peaks(corr[lo:hi])
        if not len(peaks):
            continue
        k = lo+peaks[np.argmax(corr[lo+peaks])]
        pitch[j] = rate/k
        voiced[j] = corr[k] > .65
    # A changed source never crosses an estimated unvoiced interval or uncertain edge.
    safe = voiced.copy()
    for j in range(len(safe)):
        block = slice(max(0,j-3),min(len(safe),j+4))
        safe[j] = bool(voiced[block].all()) and bool(pitch[block].max() <= 1.3*max(pitch[block].min(),1))
    sample_voiced = safe[np.minimum(np.arange(len(audio))//hop, len(safe)-1)] if len(safe) else np.zeros(len(audio),bool)
    valid = np.flatnonzero(pitch)
    contour = np.interp(np.arange(len(audio)),centers[valid],pitch[valid]) if len(valid) else np.zeros(len(audio))
    phase = np.cumsum(contour/rate)
    return contour, phase, sample_voiced


def pulse(phase, opening):
    """Derivative of a Rosenberg glottal flow, with a short smooth closure."""
    import numpy as np
    p = phase % 1
    close = (1-opening)*.6
    y = np.zeros_like(p)
    a, b = p < opening, (p >= opening) & (p < opening+close)
    y[a] = np.pi/(2*opening)*np.sin(np.pi*p[a]/opening)
    y[b] = -np.pi/(2*close)*np.sin(np.pi*(p[b]-opening)/close)
    return y


def resynthesize(audio, phase, voiced, opening=None, aspiration=0., seed=1729, rate=16000):
    """Same LPC filter, timeline and estimated pitch; copy unvoiced samples exactly.

    opening=None is an actual inverse-filter/synthesis identity control, rather
    than returning the original waveform without passing through the filter.
    """
    import numpy as np
    from scipy.signal import lfilter
    x = np.asarray(audio, dtype=np.float64)
    out, norm = np.zeros(len(x)), np.zeros(len(x))
    width, hop, context = rate//20, rate//100, rate//10
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=len(x))
    source = None if opening is None else pulse(phase, opening)+aspiration*noise
    for c in range(0,len(x),hop):
        a,b = max(0,c-width//2),min(len(x),c+width//2)
        left,right = max(0,a-context),min(len(x),b+context)
        original = x[left:right]
        filt = lpc(original)
        residual = lfilter(filt,[1.],original)
        if source is None:
            excitation = residual
        else:
            excitation = source[left:right].copy()
            excitation *= np.sqrt(np.mean(residual**2)/max(np.mean(excitation**2),1e-12))
            excitation = np.where(voiced[left:right],excitation,residual)
        rebuilt = lfilter([1.],filt,excitation)[a-left:b-left]
        window = np.maximum(np.hanning(b-a),1e-6)
        out[a:b] += rebuilt*window
        norm[a:b] += window
    out = out/np.maximum(norm,1e-12)
    out[~voiced] = x[~voiced]
    if not np.isfinite(out).all():
        raise ValueError('Non-finite source-filter view')
    return out.astype(np.float32)


def occurrence_occupancies(log_probs, ids):
    """Transcript CTC forward/backward; return one soft column per occurrence.

    Repeated phone identities remain distinct occurrences. Used only as fixed
    pooling weights, never as framewise invariance targets or hard boundaries.
    """
    import numpy as np
    labels = np.array([0]+[v for p in ids for v in (p,0)])
    t, n = len(log_probs), len(labels)
    alpha, beta = np.full((t,n),-np.inf), np.full((t,n),-np.inf)
    alpha[0,:2] = log_probs[0,labels[:2]]
    for i in range(1,t):
        for s,p in enumerate(labels):
            prev = [alpha[i-1,s]]
            if s: prev.append(alpha[i-1,s-1])
            if s>1 and p and p!=labels[s-2]: prev.append(alpha[i-1,s-2])
            alpha[i,s] = np.logaddexp.reduce(prev)+log_probs[i,p]
    z = np.logaddexp(alpha[-1,-1],alpha[-1,-2])
    if not np.isfinite(z):
        raise ValueError('Transcript cannot complete on encoder clock')
    beta[-1,-2:] = 0
    for i in range(t-2,-1,-1):
        for s,p in enumerate(labels):
            nxt = [s]
            if s+1<n: nxt.append(s+1)
            if s+2<n and labels[s+2] and labels[s+2]!=p: nxt.append(s+2)
            beta[i,s] = np.logaddexp.reduce([beta[i+1,k]+log_probs[i+1,labels[k]] for k in nxt])
    return np.exp(alpha[:,1::2]+beta[:,1::2]-z).astype(np.float32)


def cache(args):
    import numpy as np
    import onnxruntime as ort
    import torch
    import torchaudio
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    import math
    out, source = Path(args.out), Path(args.model)
    labels_path = Path(args.labels)
    labels = [json.loads(s) for s in labels_path.read_text().splitlines() if s.strip()]
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    excluded.update(r.get('speaker') for r in labels if r.get('split') != 'train')
    chosen = sorted([r for r in labels if eligible(r,excluded)],key=lambda r:(r['speaker'],r['id']))
    speaker_folds(r['speaker'] for r in chosen)
    meta = json.loads((out/'glottal-source-metadata.json').read_text())
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['sha256'] != meta['primarySha256'] or manifest['sampleRate'] != 16000 or manifest['frameMs'] != 40:
        raise ValueError('Incompatible frozen encoder')
    reference = json.loads(Path(args.reference).read_text())
    texts = {f"{a['s']}:{a['a']}":a['p'] for a in reference['ayahs']}
    token_ids = {p:i+1 for i,p in enumerate(meta['tokens'])}
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(out/'glottal-source.onnx'),options,providers=['CPUExecutionProvider'])
    clock_file=out/'training-start.json'
    if clock_file.exists():
        started=json.loads(clock_file.read_text())['started']
    else:
        started=time.time()
        clock_file.write_text(json.dumps(dict(started=started)))
    deadline = started+23*3600
    (out/'cache').mkdir(exist_ok=True)
    records = []
    for row in chosen:
        if time.time()>deadline: raise TimeoutError('One-day budget exceeded')
        wav = (labels_path.parent/row['wav']).resolve()
        if not wav.is_relative_to(labels_path.parent.resolve()): raise ValueError('Training WAV escapes label directory')
        pronunciation=''.join(texts[a] for a in row['ayahs'])
        ids=[token_ids[p] for p in pronunciation]
        key=hashlib.sha256((row['speaker']+'\0'+row['id']).encode()).hexdigest()
        receipt=out/'cache'/f'{key}.json'
        provenance=dict(primarySha256=manifest['sha256'],waveSha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
            transcriptSha256=hashlib.sha256(pronunciation.encode()).hexdigest(),recipe=1)
        if receipt.exists():
            previous=json.loads(receipt.read_text())
            if previous.get('provenance')!=provenance or previous['speaker']!=row['speaker'] or previous['id']!=row['id']:
                raise ValueError('Resumed cache input changed')
            with np.load(out/'cache'/f'{key}.npz') as cached:
                np.testing.assert_array_equal(cached['ids'],ids)
                if cached['features'].shape!=(4,previous['frames'],meta['width']): raise ValueError('Resumed cache geometry changed')
            records.append(previous)
            print(json.dumps(dict(resumed=key)),flush=True)
            continue
        rate, audio = wavfile.read(wav)
        if audio.dtype.kind in 'iu': audio=audio.astype(np.float64)/max(abs(np.iinfo(audio.dtype).min),np.iinfo(audio.dtype).max)
        else: audio=audio.astype(np.float64)
        if audio.ndim>1: audio=audio.mean(axis=1)
        if rate!=16000:
            divisor=math.gcd(int(rate),16000); audio=resample_poly(audio,16000//divisor,int(rate)//divisor)
        if len(audio)<1600 or not np.isfinite(audio).all(): raise ValueError('Invalid training waveform')
        pitch,phase,voiced=excitation_analysis(audio)
        identity=resynthesize(audio,phase,voiced)
        np.testing.assert_allclose(identity,audio,atol=2e-6,rtol=1e-5)
        views=[audio,identity,resynthesize(audio,phase,voiced,.45,.04),resynthesize(audio,phase,voiced,.75,.16)]
        features, occupancies = [], []
        for view in views:
            f=torchaudio.compliance.kaldi.fbank(torch.tensor(view,dtype=torch.float32).unsqueeze(0),
                num_mel_bins=80,dither=0,energy_floor=1e-10,high_freq=-400).numpy()
            total=max(0,((len(f)-7)//2+1)//2)
            states={s['name']:np.zeros(s['shape'],np.int64 if s['dtype']=='int64' else np.float32) for s in manifest['states']}
            hs,ls=[],[]
            for at in range(0,len(f),manifest['shift']):
                if sum(len(h) for h in hs)>=total: break
                window=np.full((1,manifest['window'],80),manifest['fbank']['padValue'],np.float32)
                piece=f[at:at+manifest['window']]; window[0,:len(piece)]=piece
                outputs=dict(zip([o.name for o in session.get_outputs()],session.run(None,{'x':window,**states})))
                hs.append(outputs[meta['featureOutput']].reshape(-1,meta['width']))
                ls.append(outputs['log_probs'].reshape(-1,len(meta['tokens'])+1))
                states={s['name']:outputs['new_'+s['name']] for s in manifest['states']}
            h,lp=np.concatenate(hs)[:total],np.concatenate(ls)[:total]
            features.append(h); occupancies.append(occurrence_occupancies(lp,ids))
        np.savez_compressed(out/'cache'/f'{key}.npz',features=np.array(features,dtype=np.float32),
            occupancies=np.array(occupancies),ids=np.array(ids,np.int64))
        records.append(dict(key=key,speaker=row['speaker'],id=row['id'],frames=len(features[0]),
            source='help-train',kind='none',split='train',voicedFraction=float(voiced.mean()),
            identityMaxError=float(np.max(np.abs(identity-audio))),provenance=provenance))
        receipt.write_text(json.dumps(records[-1],allow_nan=False))
        print(json.dumps(records[-1]),flush=True)
    (out/'training-records.json').write_text(json.dumps(records,indent=2))


def consistency(logits, occupancies):
    """Jensen-Shannon divergence of pooled nonblank identity per occurrence."""
    import torch
    identity=logits[:,:,1:].softmax(-1)
    pools=torch.einsum('vtu,vtp->vup',occupancies,identity)
    pools=pools/occupancies.sum(1).clamp_min(1e-8).unsqueeze(-1)
    mean=pools.mean(0).clamp_min(1e-8)
    return (pools*(pools.clamp_min(1e-8).log()-mean.log())).sum(-1).mean()


def fit(args):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    out=Path(args.out); meta=json.loads((out/'glottal-source-metadata.json').read_text())
    records=json.loads((out/'training-records.json').read_text())
    excluded=set(json.loads(Path(args.excluded_speakers).read_text()))
    if any(not eligible(r,excluded) for r in records): raise ValueError('Ineligible cached training speaker')
    folds=speaker_folds(r['speaker'] for r in records)
    data=[]
    for r in records:
        with np.load(out/'cache'/f"{r['key']}.npz") as cached:
            data.append((r,*(torch.tensor(cached[k]) for k in ['features','occupancies','ids'])))
    initial_w=torch.tensor(meta['initialWeight'],dtype=torch.float32)
    initial_b=torch.tensor(meta['initialBias'],dtype=torch.float32)
    criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False,reduction='mean')
    deadline=json.loads((out/'training-start.json').read_text())['started']+23*3600
    def train(rows, strength):
        head=torch.nn.Linear(meta['width'],len(meta['tokens'])+1)
        with torch.no_grad(): head.weight.copy_(initial_w); head.bias.copy_(initial_b)
        opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=0.)
        for epoch in range(8):
            for r,h,o,ids in rows:
                if time.time()>deadline: raise TimeoutError('One-day budget exceeded')
                opt.zero_grad()
                z=head(h)
                loss=criterion(z.log_softmax(-1).transpose(0,1),ids.repeat(len(h)),
                    torch.full((len(h),),h.shape[1],dtype=torch.long),torch.full((len(h),),len(ids),dtype=torch.long))
                anchor=(head.weight-initial_w).square().mean()+(head.bias-initial_b).square().mean()
                loss=loss+strength*consistency(z,o)+.01*anchor
                if not torch.isfinite(loss): raise ValueError('Non-finite paired CTC objective')
                loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),5.); opt.step()
        return head
    def validation(head, rows):
        by_speaker={}
        with torch.no_grad():
            for r,h,o,ids in rows:
                z=head(h)
                value=criterion(z.log_softmax(-1).transpose(0,1),ids.repeat(len(h)),
                    torch.full((len(h),),h.shape[1],dtype=torch.long),torch.full((len(h),),len(ids),dtype=torch.long))
                by_speaker.setdefault(r['speaker'],[]).append(float(value))
        return float(np.mean([np.mean(v) for v in by_speaker.values()]))
    # One fixed grid and update budget, chosen entirely in speaker-separated training folds.
    scores=[]
    for strength in [.1,1.]:
        values=[]
        for fold in range(3):
            train_rows=[d for d in data if folds[d[0]['speaker']]!=fold]
            val_rows=[d for d in data if folds[d[0]['speaker']]==fold]
            values.append(validation(train(train_rows,strength),val_rows))
        scores.append(dict(strength=strength,foldCtc=values,mean=float(np.mean(values))))
        print(json.dumps(scores[-1]),flush=True)
    selected=min(scores,key=lambda r:(r['mean'],r['strength']))['strength']
    invariant,control=train(data,selected),train(data,0.)
    def arm(head): return dict(weight=head.weight.detach().tolist(),bias=head.bias.detach().tolist())
    model={k:meta[k] for k in ['schema','inventory','tokens','width','featureOutput','primarySha256']}
    model.update(invariant=arm(invariant),control=arm(control))
    (out/'glottal-source-readouts.json').write_text(json.dumps(model,allow_nan=False))
    report=dict(records=len(data),speakers=len(folds),folds=folds,selection=scores,selectedStrength=selected,
        epochs=8,learningRate=1e-4,anchor=.01,views=['original','identity','open-.45-aspiration-.04','open-.75-aspiration-.16'],
        invariantTrainingCtc=validation(invariant,data),controlTrainingCtc=validation(control,data))
    (out/'training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    files=[]
    for name in ['glottal-source-readouts.json','glottal-source.onnx']:
        b=(out/name).read_bytes(); files.append(dict(file=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    (out/'glottal-source-asset.json').write_text(json.dumps(dict(schema=1,files=files)))
    print(json.dumps(report),flush=True)


def bundle(args):
    """Install unchanged primary assets and verify the complete side bundle.

    This writes only the chosen export directory. The supplied source manifest
    and auxiliary teacher files are copied verbatim, not re-authored.
    """
    import shutil
    import numpy as np
    import onnxruntime as ort
    source,out,destination=Path(args.model),Path(args.out),Path(args.bundle_dir)
    if destination.resolve().is_relative_to(source.resolve()) or destination.resolve()==out.resolve():
        raise ValueError('Bundle must be separate from source and training cache')
    manifest=json.loads((source/'manifest.json').read_text())
    primary=(source/manifest['file']).read_bytes()
    if hashlib.sha256(primary).hexdigest()!=manifest['sha256']: raise ValueError('Primary checksum mismatch')
    fitted=json.loads((out/'glottal-source-readouts.json').read_text())
    if any(fitted[k]!=manifest[v] for k,v in [('primarySha256','sha256'),('tokens','tokens'),('inventory','inventory')]):
        raise ValueError('Readouts do not belong to primary graph')
    descriptor=json.loads((out/'glottal-source-asset.json').read_text())
    if descriptor['schema']!=1 or [f['file'] for f in descriptor['files']]!=['glottal-source-readouts.json','glottal-source.onnx']:
        raise ValueError('Invalid bundle declaration')
    for file in descriptor['files']:
        b=(out/file['file']).read_bytes()
        if len(b)!=file['bytes'] or hashlib.sha256(b).hexdigest()!=file['sha256']: raise ValueError('Side asset checksum mismatch')
    for arm in [fitted['invariant'],fitted['control']]:
        w,b=np.array(arm['weight']),np.array(arm['bias'])
        if w.shape!=(len(manifest['tokens'])+1,fitted['width']) or b.shape!=(len(w),) or not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError('Invalid phone readout geometry')
    options=ort.SessionOptions(); options.intra_op_num_threads=1
    a=ort.InferenceSession(str(source/manifest['file']),options,providers=['CPUExecutionProvider'])
    b=ort.InferenceSession(str(out/'glottal-source.onnx'),options,providers=['CPUExecutionProvider'])
    names=[o.name for o in a.get_outputs()]
    states={s['name']:np.zeros(s['shape'],np.int64 if s['dtype']=='int64' else np.float32) for s in manifest['states']}
    rng=np.random.default_rng(1729)
    for _ in range(8):
        feeds={'x':rng.normal(size=(1,manifest['window'],manifest['fbank']['mels'])).astype(np.float32),**states}
        x,y=a.run(names,feeds),b.run(names+[fitted['featureOutput']],feeds)
        for name,u,v in zip(names,x,y): np.testing.assert_array_equal(u,v,err_msg='Bundle changed primary '+name)
        if y[-1].size!=x[names.index('log_probs')].size//(len(manifest['tokens'])+1)*fitted['width']: raise ValueError('Extractor frame clock mismatch')
        states={s['name']:y[names.index('new_'+s['name'])] for s in manifest['states']}
    shutil.copytree(source,destination,dirs_exist_ok=True)
    for name in ['glottal-source.onnx','glottal-source-readouts.json','glottal-source-asset.json']:
        shutil.copy2(out/name,destination/name)
    if (destination/'manifest.json').read_bytes()!=(source/'manifest.json').read_bytes(): raise ValueError('Primary manifest changed')
    # Copying the entire original directory retains all existing insertion-teacher external data.
    def checksum(path):
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1<<20),b''): digest.update(block)
        return digest.digest()
    for path in source.rglob('*'):
        if path.is_file() and checksum(path)!=checksum(destination/path.relative_to(source)):
            raise ValueError('Copied primary asset differs')
    for file in descriptor['files']:
        if checksum(out/file['file'])!=checksum(destination/file['file']): raise ValueError('Copied side asset differs')
    print(json.dumps(dict(bundle=str(destination),primarySha256=manifest['sha256'],parityWindows=8,verified=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['cache','fit','bundle'])
    parser.add_argument('--model'); parser.add_argument('--labels'); parser.add_argument('--reference'); parser.add_argument('--bundle-dir')
    parser.add_argument('--excluded-speakers',required=True); parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.stage=='cache' and not all([args.model,args.labels,args.reference]): parser.error('cache requires model, labels and reference')
    if args.stage=='bundle' and not all([args.model,args.bundle_dir]): parser.error('bundle requires model and bundle-dir')
    {'cache':cache,'fit':fit,'bundle':bundle}[args.stage](args)
