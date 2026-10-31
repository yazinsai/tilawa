"""Read-only operational status for a cached decode, including preempted workers."""
import json
from pathlib import Path
import modal
app = modal.App('tilawa-sparse-status')
vol = modal.Volume.from_name('tilawa-lab')
@app.function(image=modal.Image.debian_slim(), volumes={'/lab': vol})
def status(run: str):
    root = Path('/lab/runs') / run
    rows = [json.loads(p.read_text()) for p in root.glob('*/status.json')]
    return {'run': run, 'directories': len(list(root.iterdir())), 'finished': len(rows),
            'failed': [r for r in rows if r['code']]}
@app.local_entrypoint()
def main(run: str):
    print(json.dumps(status.remote(run)), flush=True)
