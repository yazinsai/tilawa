"""Run a matched frozen-encoder refit on an explicitly authorized training list.

Run on remote compute, with the pinned student training dependencies available:

    PYTHONPATH=modal:tools:$PYTHONPATH python3 modal/student_prefix_refit.py \
        --checkpoint <shipped-source-ckpt.pt> --recordings <training.json> \
        --deployed-manifest <shipped-manifest.json> --output <run-directory> --epochs 3 --export

The JSON list supplies speaker, source (authorized-transcribed/help-train-none),
held_out=false, transcript_kind=performed, audio and phones for each recording.
Help-train also requires label=none. Phones are CTC occurrence sequences, not
duration-expanded notation. Audio paths resolve relative to the list.
Scripted mistakes, repeats and canonical-only guesses are not transcripts.
The caller must resolve release exclusions before preparing this list.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from student_prefix import refit_pair, streaming_features


def refit(checkpoint, items, output, deployed_manifest, *, epochs=3, seed=0, lr=0.001,
          confidence=0.95, prefixes_per_recording=4, prefix_weight=1.0, export=False):
    from student import TOKENS, phoneme_ids
    from student_export import export_onnx, fbank, manifest, stream_onnx
    from student_train import build_model

    started = time.monotonic()
    checkpoint, items, output = Path(checkpoint), Path(items), Path(output)
    deployed_manifest = Path(deployed_manifest)
    geometry = json.loads(deployed_manifest.read_text())
    chunk, left = geometry['chunk'], geometry['leftContext']
    if (geometry['tokens'] != TOKENS or geometry['blank'] != 0 or geometry['frameMs'] != 40
            or geometry['sampleRate'] != 16000 or geometry['window'] != 2 * chunk + 13
            or geometry['shift'] != 2 * chunk or geometry['fbank']['mels'] != 80
            or chunk <= 0 or chunk % 2 or left <= 0):
        raise ValueError('Deployed model must match the existing student inventory, frontend and clock')
    if (output / 'control').exists() or (output / 'prefix').exists():
        raise ValueError('Use a new output directory; do not overwrite a matched run')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build_model(ck['cfg']['size']).eval()
    model.load_state_dict(ck['model'], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    recordings, examples = [], []
    for item in json.loads(items.read_text()):
        # Fail closed before reading or encoding any ineligible audio.
        if (item.get('source') not in ('authorized-transcribed', 'help-train-none')
                or item.get('held_out') is not False or not item.get('speaker')
                or item.get('transcript_kind') != 'performed'
                or (item['source'] == 'help-train-none' and item.get('label') != 'none')):
            raise ValueError('Every item must carry explicit training eligibility and a performed transcript')
        audio = items.parent / item['audio']
        feats = fbank(audio.read_bytes())
        x = streaming_features(model, feats, chunk, left)
        recordings.append(dict(item, features=x, target=phoneme_ids(item['phones'])))
        if len(examples) < 3:
            examples.append((feats, x))
        if time.monotonic() - started > 24 * 3600:
            raise TimeoutError('One-day cache/refit budget exhausted')
    # The optimizer sees only this copy of the original phone head.
    head = copy.deepcopy(model.ctc_output)
    for p in head.parameters():
        p.requires_grad_(True)
    selected, report = refit_pair(head, recordings, epochs=epochs, lr=lr, seed=seed, chunk=chunk,
                                 confidence=confidence, prefixes_per_recording=prefixes_per_recording,
                                 prefix_weight=prefix_weight, budget_seconds=24 * 3600 - (time.monotonic() - started))
    report.update(source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  recordings_sha256=hashlib.sha256(items.read_bytes()).hexdigest(),
                  deployed_manifest_sha256=hashlib.sha256(deployed_manifest.read_bytes()).hexdigest(),
                  chunk=chunk, left=left, frame_ms=40, elapsed_s=time.monotonic() - started,
                  confidence=confidence, prefixes_per_recording=prefixes_per_recording,
                  prefix_weight=prefix_weight, preservation_weight=1.0, lr=lr)
    for name, selection in selected.items():
        model.ctc_output.load_state_dict(selection['state_dict'])
        dest = output / name
        dest.mkdir(parents=True)
        result = dict(ck, model=copy.deepcopy(model.state_dict()),
                      cfg=dict(ck['cfg'], name=name, prefix_refit=report),
                      state=dict(epoch=selection['epoch'], step=selection['epoch'] * report['training_recordings']))
        # Original optimizer/schedule state does not describe this head refit.
        result.pop('opt', None)
        result.pop('sched', None)
        torch.save(result, dest / 'ckpt.pt')
        if export:
            path = dest / 'model.onnx'
            states = export_onnx(model, str(path), chunk, left)
            rows = []
            for feats, features in examples:
                with torch.no_grad():
                    expected = model.ctc_output(features).log_softmax(-1).numpy()
                got = stream_onnx(str(path), feats, states, chunk)[:len(features)]
                diff = float(abs(got - expected).max())
                agree = float((got.argmax(-1) == expected.argmax(-1)).mean())
                rows.append(dict(frames=len(features), max_abs_log_prob_diff=diff, top1_agreement=agree))
                if diff > 0.001 or agree != 1:
                    raise ValueError(f'Export parity failed: {rows[-1]}')
            (dest / 'manifest.json').write_text(json.dumps(manifest(name, states, chunk, left, str(path), result), indent=2))
            (dest / 'parity.json').write_text(json.dumps(rows, indent=2))
    (output / 'refit.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint', 'recordings', 'output', 'deployed-manifest'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--export', action='store_true')
    args = vars(parser.parse_args())
    args['items'] = args.pop('recordings')
    print(json.dumps(refit(**args), indent=2))
