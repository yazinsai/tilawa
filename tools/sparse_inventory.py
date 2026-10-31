import json,os
from pathlib import Path
root=Path('/lab/data/recordings')
report={'recordingFiles':[str(p.relative_to(root)) for p in root.rglob('*.json')],'selfCorrections':[]}
for p in root.rglob('*.json'):
 try: d=json.loads(p.read_text())
 except Exception: continue
 if 'self_correction' in json.dumps(d): report['selfCorrections'].append({'path':str(p),'data':d})
Path(os.environ['LAB_OUT'],'inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=1)); print(json.dumps({'recordingFiles':len(report['recordingFiles']),'selfCorrectionFiles':len(report['selfCorrections'])}))
