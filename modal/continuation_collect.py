import json,hashlib
from pathlib import Path
import modal
app=modal.App('tilawa-continuation-collect')
lab=modal.Volume.from_name('tilawa-lab');student=modal.Volume.from_name('tilawa-student')
@app.function(image=modal.Image.debian_slim(),volumes={'/lab':lab,'/student':student})
def collect(kind: str, run: str):
    if kind=='training':
        root=Path('/student/runs')/run
        rows=[json.loads(s) for s in (root/'log.jsonl').read_text().splitlines()]
        keys=(root/'manifest-train.json').read_bytes()
        return dict(log=rows,poolSha256=hashlib.sha256(keys).hexdigest(),poolKeys=json.loads(keys))
    root=Path('/lab/runs')/run
    return {str(p.relative_to(root)):json.loads(p.read_text()) for p in root.glob('*/'+('trace.json' if kind=='trace' else 'status.json'))}
@app.local_entrypoint()
def main(kind: str, run: str, dest: str):
    obj=collect.remote(kind,run)
    Path(dest).write_text(json.dumps(obj,ensure_ascii=False,indent=1))
    print(json.dumps(dict(kind=kind,run=run,entries=len(obj)),ensure_ascii=False))
