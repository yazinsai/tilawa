"""Train causal speaker/channel FiLM using only gold-excluded clean-match training clusters.

Different clips from the same voice cluster share one shipped augmentation draw.
No practice or release audio is read by this entry point.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio
import student_train as T
from student_session import SessionStudent, stable_swoosh_training
from student import exclusions, phoneme_ids, ctc_collapse, mix_noise, soft_target_loss


def pool(out, hours=20):
    import pyarrow.parquet as pq
    cache = out / 'pool.pt'
    gold_path = Path('/root/gold-per-v1-exclude.txt')
    if not gold_path.exists():
        gold_path = Path('/app/docs/benchmarks/gold-per-v1-exclude.txt')
    gold = exclusions(gold_path.read_text())
    fingerprint = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    if cache.exists():
        saved = torch.load(cache, weights_only=False)
        if saved['gold'] != fingerprint:
            raise ValueError('Changed exclusions')
        return saved['clips']
    files = sorted(Path('/student/s1').glob('*.parquet'))
    metadata, excluded = [], set()
    for fi, path in enumerate(files):
        rows = pq.read_table(path, columns=['shard','row','bucket','split','cluster','n_samples']).to_pylist()
        for ri, r in enumerate(rows):
            if (r['shard'], r['row']) in gold:
                excluded.add(r['cluster'])
            metadata.append((fi, ri, r))
    candidates = [(fi, ri, r) for fi,ri,r in metadata if r['cluster'] not in excluded and
        r['bucket'] == 'clean-match' and r['split'] == 'train' and 16000 <= r['n_samples'] <= 30 * 16000]
    groups = {}
    for item in candidates:
        groups.setdefault(item[2]['cluster'], []).append(item)
    candidates = [x for x in candidates if len(groups[x[2]['cluster']]) > 1]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(candidates))
    selected, duration = [], 0
    for i in order:
        selected.append(candidates[i]); duration += candidates[i][2]['n_samples'] / 16000
        if duration >= hours * 3600:
            break
    by = {}
    for fi,ri,r in selected:
        by.setdefault(fi, []).append(ri)
    clips = []
    for fi, idx in by.items():
        clips.extend(pq.read_table(files[fi], columns=['shard','row','cluster','ref','hyp','flac','n_frames','top5_ids','top5_p']).take(idx).to_pylist())
    counts = {}
    for c in clips:
        counts[c['cluster']] = counts.get(c['cluster'], 0) + 1
    clips = [c for c in clips if counts[c['cluster']] > 1]
    audit = dict(clips=len(clips), clusters=len(set(c['cluster'] for c in clips)), goldExcludedClusters=len(excluded),
        hours=duration/3600, split='train', bucket='clean-match', approximateSpeakerIdentity=True)
    if not clips:
        raise ValueError('No eligible same-reader pairs')
    (out/'data.json').write_text(json.dumps(audit, indent=1))
    torch.save(dict(clips=clips, gold=fingerprint), cache)
    print(json.dumps(dict(event='pool', **audit)), flush=True)
    return clips


def feature(clip, seed, cfg, aug):
    rng = np.random.default_rng(seed)
    wave, sr = sf.read(io.BytesIO(clip['flac']), dtype='float32')
    if sr != 16000 or wave.ndim != 1:
        raise ValueError('Expected mono training audio at 16 kHz')
    if rng.random() < cfg['rir_p']:
        wave = T.reverb(wave, T.pick(aug['rir'], rng))
    if rng.random() < cfg['noise_p']:
        wave = mix_noise(wave, T.pick(aug['noise'], rng), rng.uniform(*cfg['snr_db']), rng)
    wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
    feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave))[None],
        num_mel_bins=80, sample_frequency=16000, dither=0.0, energy_floor=1e-10, high_freq=-400)
    return T.spec_augment(feats, rng) if cfg['spec_aug'] else feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/student/prior-runs/spliced-clean/ckpt.pt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--max-seconds', type=int, default=3000)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ck = torch.load(args.source, map_location='cpu', weights_only=False)
    cfg = {**T.DEFAULTS, **ck['cfg']}
    parent = T.build_model(cfg['size'])
    parent.load_state_dict(ck['model']); parent.eval(); parent.streaming_eval(16,128)
    stable_swoosh_training()
    model = SessionStudent(parent).cuda().eval(); model.streaming_eval(16,128)
    print(json.dumps(dict(event='architecture', size=cfg['size'], norms=len(model.films),
        frameEncoder=str(model.frame_encoder), cfg=cfg)), flush=True)
    clips = pool(out)
    if args.prepare_only:
        return
    aug = {k: dict(np.load(f'/student/aug/{k}.npz')) for k in ('rir','noise')}
    groups = {}
    for i,c in enumerate(clips):
        groups.setdefault(c['cluster'], []).append(i)
    opt = torch.optim.AdamW([dict(params=model.acoustic.parameters(), lr=0.00003),
        dict(params=model.films.parameters(), lr=0.0003), dict(params=model.frame_encoder.parameters(), lr=0.0003)])
    start = 0
    source_hash = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
    if (out/'ckpt.pt').exists():
        saved = torch.load(out/'ckpt.pt', weights_only=False)
        if saved['sourceSha256'] != source_hash or saved['totalSteps'] != args.steps:
            raise ValueError('Changed training recipe on resume')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['opt']); start = saved['step']
        rng.bit_generator.state = saved['rng']; torch.set_rng_state(saved['torchRng']); torch.cuda.set_rng_state_all(saved['cudaRng'])
    started = time.time()
    log = (out/'log.jsonl').open('a')
    for step in range(start, args.steps):
        opt.zero_grad(set_to_none=True)
        i = int(rng.integers(len(clips)))
        peers = [j for j in groups[clips[i]['cluster']] if j != i]
        j = int(rng.choice(peers)); seed = int(rng.integers(2**32))
        x = feature(clips[i], seed, cfg, aug).cuda()
        context = feature(clips[j], seed, cfg, aug).cuda()
        # Prefix duration varies from one second to the peer clip's entire past.
        context = context[:int(rng.integers(min(100, len(context)), len(context) + 1))]
        real = ((len(x)-7)//2+1)//2
        # Match EOS context to deployment; padding never contributes to a loss.
        padded = 45 + (max(1, (real+7)//8)-1)*32
        x = F.pad(x, (0,0,0,max(0,padded-len(x))), value=-23.0259)[None]
        lens = torch.tensor([x.shape[1]], device='cuda')
        features = model.features(context[None], torch.tensor([len(context)],device='cuda'))
        embedding = features.mean(0)
        zeroed = step % 4 == 0
        if zeroed:
            embedding = embedding * 0
        lp, _ = model(x, lens, embedding)
        lp = lp[:, :real].float()
        targets = torch.tensor(ctc_collapse(phoneme_ids(clips[i]['ref'])), device='cuda', dtype=torch.long)
        ctc = F.ctc_loss(lp.transpose(0,1), targets, torch.tensor([real],device='cuda'),
            torch.tensor([len(targets)],device='cuda'), reduction='sum', zero_infinity=True)/real
        ids = torch.from_numpy(np.frombuffer(clips[i]['top5_ids'],np.uint8).copy().reshape(-1,5)).long().cuda()
        probs = torch.from_numpy(np.frombuffer(clips[i]['top5_p'],np.float16).copy().astype(np.float32).reshape(-1,5)).cuda()
        idx = torch.arange(real,device='cuda').clamp(max=len(ids)-1)
        kd = soft_target_loss(lp, ids[idx][None], probs[idx][None], torch.ones(1,real,device='cuda',dtype=torch.bool))
        kl = lp.sum()*0
        if zeroed:
            with torch.no_grad():
                anchor,_ = model.frozen(x,lens)
            kl = F.kl_div(lp,anchor[:,:real].exp(),reduction='sum')/real
        loss = ctc + kd + 10*kl
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite objective')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        opt.step()
        stop = time.time()-started >= args.max_seconds
        if (step+1)%50 == 0:
            row = dict(step=step+1,seconds=time.time()-started,ctc=float(ctc.detach()),kd=float(kd.detach()),kl=float(kl.detach()),
                embeddingNorm=float(embedding.detach().norm()),filmNorm=float(sum(p.detach().square().sum() for p in model.films.parameters()).sqrt()))
            print(json.dumps(row),flush=True); log.write(json.dumps(row)+'\n'); log.flush()
        if (step+1)%250==0 or step+1==args.steps or stop:
            torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),step=step+1,totalSteps=args.steps,cfg=cfg,
                rng=rng.bit_generator.state,torchRng=torch.get_rng_state(),cudaRng=torch.cuda.get_rng_state_all(),sourceSha256=source_hash),out/'ckpt.pt.tmp')
            (out/'ckpt.pt.tmp').replace(out/'ckpt.pt')
        if stop:
            break
    print(json.dumps(dict(event='finished',steps=step+1,complete=step+1>=args.steps)),flush=True)


if __name__=='__main__':
    main()
