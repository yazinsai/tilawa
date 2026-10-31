"""Fixed 75/25 interpolation of the shipped and microphone-adapted student."""
import json
from pathlib import Path
import modal
from student_train import image, vol
app=modal.App('tilawa-listener-blend',image=image.add_local_python_source('student_train'))
@app.function(cpu=4,memory=16384,volumes={'/student':vol},timeout=600)
def blend():
    import hashlib
    import torch
    base_path=Path('/student/runs/spliced-clean/ckpt.pt')
    adapted_path=Path('/student/runs/channel-adapt-v1/ckpt.pt')
    base=torch.load(base_path,map_location='cpu',weights_only=False)
    adapted=torch.load(adapted_path,map_location='cpu',weights_only=False)
    assert base['model'].keys()==adapted['model'].keys()
    mixed={}
    for name,a in base['model'].items():
        b=adapted['model'][name]
        assert a.shape==b.shape and a.dtype==b.dtype
        mixed[name]=torch.lerp(a,b,0.25) if a.is_floating_point() else a.clone()
        assert torch.isfinite(mixed[name]).all()
    out=Path('/student/runs/channel-blend25-v1');out.mkdir(exist_ok=True)
    torch.save({'model':mixed,'cfg':base['cfg'],'state':base['state']},out/'ckpt.pt')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    note={'base':'spliced-clean','baseSha256':sha(base_path),'adapted':'channel-adapt-v1','adaptedSha256':sha(adapted_path),'adaptedWeight':0.25,'checkpointSha256':sha(out/'ckpt.pt'),'tensors':len(mixed)}
    (out/'blend.json').write_text(json.dumps(note,indent=1));vol.commit();return note
@app.local_entrypoint()
def main():
    print(blend.remote())
