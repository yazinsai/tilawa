"""Read-only context intervention on engine-requested windows.

The disputed raw central interval stays bit-identical. Only its immediate
temporal readout context changes. This is a premise diagnostic, not a scorer.
No input from this utility participates in model/checkpoint selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

from student_crossed_context import alignment, sample_path, occurrences, temporal


def emissions(head, features, model):
    import numpy as np
    x = (np.asarray(features, dtype='float32')-model['center'])/model['scale']
    hidden = np.tanh(temporal(x) @ np.asarray(head['hidden']).T + head['hiddenBias'])
    logits = hidden @ np.asarray(head['output']).T + head['outputBias']
    peak = logits.max(1, keepdims=True)
    return logits-peak-np.log(np.exp(logits-peak).sum(1, keepdims=True))


def disputed_pairs(written, rival):
    """Minimum-edit mapping; every changed written consonant can be probed.

    An omitted consonant has blank as its rival. Vowel-only and insertion-only
    edits supply no central consonant premise evidence.
    """
    import numpy as np
    cost = np.zeros((len(written)+1, len(rival)+1), dtype=int)
    cost[:, 0] = np.arange(len(written)+1)
    cost[0] = np.arange(len(rival)+1)
    for i, p in enumerate(written, 1):
        for j, q in enumerate(rival, 1):
            cost[i, j] = min(cost[i-1, j]+1, cost[i, j-1]+1, cost[i-1, j-1]+(p != q))
    i, j, result = len(written), len(rival), []
    while i or j:
        if i and j and cost[i, j] == cost[i-1, j-1]+(written[i-1] != rival[j-1]):
            if written[i-1] != rival[j-1]:
                result.append((i-1, written[i-1], rival[j-1]))
            i, j = i-1, j-1
        elif i and cost[i, j] == cost[i-1, j]+1:
            result.append((i-1, written[i-1], 0))
            i -= 1
        else:
            j -= 1
    return result[::-1]


def exchange(features, a, b, donor, radius=3):
    """Use real donor flank frames; never interpolate or replace the target core."""
    import numpy as np
    out = np.array(features, dtype='float32', copy=True)
    left, right = donor['parts'][0][0], donor['parts'][2][0]
    nl, nr = min(radius, a, len(left)), min(radius, len(out)-b, len(right))
    if nl:
        out[a-nl:a] = left[-nl:]
    if nr:
        out[b:b+nr] = right[:nr]
    np.testing.assert_array_equal(out[a:b], features[a:b])
    return out, nl+nr


def probe(args):
    import numpy as np
    model = json.loads(Path(args.model).read_text())
    training = [json.loads(s) for s in Path(args.training).read_text().splitlines() if s.strip()]
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    if any(r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or
           r['speaker'] in excluded for r in training):
        raise ValueError('Context donors must be authorized training speakers')
    rows = []
    for r in training:
        if r['kind'] == 'none':
            aligned = alignment(r['logProbs'], r['pronunciation'])
            if aligned is not None:
                rows.append(dict(r, aligned=aligned, x=np.asarray(r['features'], dtype='float32')))
    rng = np.random.default_rng(1729)
    items = occurrences(rows, model['tokens'], rng)
    results = []
    for name in args.diagnostic:
        for entry in json.loads(Path(name).read_text())['comparisons']:
            c, comparison = entry['candidate'], entry['comparison']
            written = [model['tokens'].index(p)+1 for p in c['written']]
            rival = [model['tokens'].index(p)+1 for p in c['rival']]
            aligned = alignment(entry['logProbs'], written)
            tests = []
            if aligned is not None and comparison is not None:
                features = np.asarray(entry['features'], dtype='float32')
                baseline = emissions(model['control'], features, model)
                for pos, phone, other in disputed_pairs(written, rival):
                    context = (written[pos-1] if pos else 0, written[pos+1] if pos+1 < len(written) else 0)
                    pool = [v for v in items if v['phone'] == phone and v['context'] != context and
                            len(v['parts'][0][0])+len(v['parts'][2][0])]
                    # Stable recording-balanced pool, at most sixteen independent takes.
                    pool.sort(key=lambda v: hashlib.sha256(v['recording'].encode()).hexdigest())
                    unique = {v['recording']: v for v in pool}
                    donors = list(unique.values())[:16]
                    if len(donors) < 3 or len({v['speaker'] for v in donors}) < 3:
                        continue
                    paths = []
                    for _ in range(16):
                        path = sample_path(aligned, rng)
                        at = np.flatnonzero(path == 2*pos+1)
                        a, b = int(at[0]), int(at[-1])+1
                        original = float((baseline[a:b, phone]-baseline[a:b, other]).mean())
                        changed = []
                        for donor in donors:
                            intervention, n = exchange(features, a, b, donor)
                            if n:
                                lp = emissions(model['control'], intervention, model)
                                changed.append(float((lp[a:b, phone]-lp[a:b, other]).mean()))
                        median = float(np.median(changed)) if changed else None
                        paths.append(dict(fromFrame=c['fromFrame']+a, toFrame=c['fromFrame']+b,
                            original=original, donorMedian=median,
                            reversal=median is not None and original < 0 < median))
                    tests.append(dict(position=pos, written=phone, rival=other, donorRecordings=len(donors),
                        donorSpeakers=len({v['speaker'] for v in donors}), paths=paths,
                        sensitive=sum(p['reversal'] for p in paths) > len(paths)/2))
            results.append(dict(diagnostic=name, candidate=c, tests=tests,
                sensitive=bool(comparison and comparison['controlWritten'] < comparison['controlRival'] and
                               any(t['sensitive'] for t in tests))))
    report = dict(schema=1, draws=16, maximumDonorRecordings=16, results=results,
                  sensitiveComparisons=sum(r['sensitive'] for r in results))
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(comparisons=len(results), sensitiveComparisons=report['sensitiveComparisons'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Frozen crossed-context-heads.json')
    p.add_argument('--training', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--diagnostic', action='append', required=True, help='SDK requested-window diagnostic JSON')
    p.add_argument('--out', required=True)
    probe(p.parse_args())
