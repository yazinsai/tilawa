"""Quantize an exported adaptation on Modal and publish a separate lab candidate."""
import json
from pathlib import Path
import modal
image = modal.Image.debian_slim(python_version="3.12").pip_install("onnx==1.19.0", "onnxruntime==1.22.1")
app = modal.App("tilawa-phone-publish", image=image)
student = modal.Volume.from_name("tilawa-student")
lab = modal.Volume.from_name("tilawa-lab")
@app.function(cpu=8, memory=32768, volumes={"/student":student,"/lab":lab}, timeout=3600)
def publish(run: str):
    import hashlib
    from shutil import copyfile
    from onnxruntime.quantization import QuantType, quantize_dynamic
    src=Path('/student/runs')/run/'onnx'
    out=Path('/lab/data/models/paired-consonants-int8')
    out.mkdir(parents=True,exist_ok=True)
    quantize_dynamic(str(src/'model.onnx'),str(out/'model.onnx'),op_types_to_quantize=['MatMul'],weight_type=QuantType.QInt8)
    data=(out/'model.onnx').read_bytes()
    m=json.loads((src/'manifest.json').read_text())
    m.update(id='paired-consonants-int8',format='onnx-int8',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),quantizedFrom=m['sha256'])
    (out/'manifest.json').write_text(json.dumps(m,indent=1))
    copyfile(src/'parity.json',out/'parity.json')
    lab.commit()
    return m
@app.local_entrypoint()
def main(run: str='paired-consonants-v1'):
    print(publish.remote(run))
