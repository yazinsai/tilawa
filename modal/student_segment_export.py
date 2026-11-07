"""Bundle the frozen primary, learned segment scorer, and existing auxiliary teacher.

Only a new export directory is written. Supplied models and manifests remain
read-only. A complete verified bundle is required before declaring deployment.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def export(args):
    model,features,scorer,teacher,out=map(Path,(args.model,args.features,args.scorer,args.teacher,args.out))
    manifest=json.loads((model/'manifest.json').read_text())
    head=json.loads((scorer/'segment-scorer.json').read_text())
    if digest(model/manifest['file'])!=manifest['sha256']:
        raise ValueError('Primary model checksum differs from its supplied manifest')
    if digest(features/'segment-encoder.onnx')!=head['encoderSha256']:
        raise ValueError('Feature graph differs from the frozen training encoder')
    if head['tokens']!=manifest['tokens'] or head['inventory']!=manifest['inventory']:
        raise ValueError('Scorer and primary inventories differ')
    # Preserve the folded teacher asset exactly, rather than shipping a primary-only baseline.
    registry=(ROOT/'packages/sdk/src/registry.ts').read_text()
    for name in ('teacher.onnx.data',):
        match=re.search(r'file:\s*"'+re.escape(name)+r'",\s*bytes:\s*(\d+),\s*sha256:\s*"([0-9a-f]+)"',registry)
        if not match or (teacher/name).stat().st_size!=int(match[1]) or digest(teacher/name)!=match[2]:
            raise ValueError('Teacher weights differ from the existing declared bundle: '+name)
    out.mkdir(parents=True,exist_ok=True)
    sources={manifest['file']:model/manifest['file'],'segment-encoder.onnx':features/'segment-encoder.onnx',
             'segment-scorer.json':scorer/'segment-scorer.json',
             'teacher.onnx':teacher/'teacher.onnx','teacher.onnx.data':teacher/'teacher.onnx.data'}
    for name,path in sources.items():
        shutil.copyfile(path,out/name)
    manifest['segmentPresence']={'file':'segment-scorer.json','sha256':digest(out/'segment-scorer.json'),
                                'encoder':{'file':'segment-encoder.onnx','sha256':digest(out/'segment-encoder.onnx')}}
    part=lambda name: {'file':name,'bytes':(out/name).stat().st_size,'sha256':digest(out/name)}
    manifest['insertionTeacher']={'inventory':manifest['inventory'],**part('teacher.onnx'),
                                 'externalData':[part('teacher.onnx.data')]}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=1)+'\n')
    report={name:{'bytes':(out/name).stat().st_size,'sha256':digest(out/name)} for name in [*sources,'manifest.json']}
    print(json.dumps(report,indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('model','features','scorer','teacher','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--report')
    export(parser.parse_args())
