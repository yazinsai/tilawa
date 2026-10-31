import json
from pathlib import Path
import modal
import student_export as E
import phone_publish as P
app=modal.App('tilawa-continuation-export')
app.include(E.app)
app.include(P.app)
@app.local_entrypoint()
def main(arm: str):
    assert arm in ('clean','phone')
    base=Path('/Users/rock/ai/projects/tilawa/.scratch/help')
    labels=[json.loads(s) for s in (base/'eval-labels.jsonl').read_text().splitlines()]
    picks=[l for l in labels if not l['exclude']][::max(1,len(labels)//6)][:6]
    result=E.export.remote('continuation-'+arm+'-v2',[(l['id'],(base/l['wav']).read_bytes()) for l in picks])
    out=Path('.scratch/lab/experiments')
    (out/('continuation-'+arm+'-export.json')).write_text(json.dumps(result,ensure_ascii=False,indent=1))
    assert all(x['edits']==0 for x in result['parity']['clips']),result['parity']
    m=P.publish.remote('continuation-'+arm+'-v2','continuation-'+arm+'-int8')
    (out/('continuation-'+arm+'-manifest.json')).write_text(json.dumps(m,ensure_ascii=False,indent=1))
    print(json.dumps(dict(arm=arm,manifest=m,parity=result['parity']),ensure_ascii=False),flush=True)
