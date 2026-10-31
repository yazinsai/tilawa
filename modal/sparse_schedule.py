"""Read-only check of actual sparse exposure under the frozen 200-batch plan."""
import json
from pathlib import Path
import modal
import student_train as S

app = modal.App('tilawa-sparse-schedule', image=S.image.add_local_python_source('student_train'))

@app.function(cpu=4, memory=16384, volumes={'/student': S.vol}, timeout=900)
def inspect():
    import numpy as np
    import pyarrow.parquet as pq
    keys = json.loads(Path('/student/runs/sparse-consonants-v1/manifest-train.json').read_text())
    other = json.loads(Path('/student/runs/sparse-random-v1/manifest-train.json').read_text())
    assert keys == other
    wanted = set(keys)
    lengths = {}
    for path in sorted(Path('/student/s1').glob('*.parquet')):
        rows = pq.read_table(path, columns=['shard', 'row', 'n_samples']).to_pydict()
        for shard, row, n in zip(rows['shard'], rows['row'], rows['n_samples']):
            key = f'{shard}:{row}'
            if key in wanted:
                lengths[key] = n
    assert set(lengths) == wanted
    plan = S.batches(np.array([lengths[k] for k in keys]), 800., 0)
    seen = {keys[i]: batch for batch, indices in enumerate(plan[:200], 1) for i in indices}
    frozen = json.loads(Path('/student/runs/sparse-mining-v1/verified-manifest.json').read_text())
    return {'plannedBatches': len(plan), 'executedBatches': 200, 'seenClips': len(seen),
            'arms': {arm: {'seen': {k: seen[k] for k in frozen[arm] if k in seen},
                           'outsideBudget': [k for k in frozen[arm] if k not in seen]}
                     for arm in ('selected', 'random')}}

@app.local_entrypoint()
def main():
    result = inspect.remote()
    Path('.scratch/lab/experiments/sparse-exposure.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
