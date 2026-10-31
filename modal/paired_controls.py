"""Evaluation-only SDK controls restricted to three human-confirmed mistakes."""
import json
from pathlib import Path
import modal
app = modal.App('tilawa-paired-controls')
if modal.is_local():
    import lab_run
    app.include(lab_run.app)
vol = modal.Volume.from_name('tilawa-lab')
@app.function(image=modal.Image.debian_slim(), volumes={'/lab':vol})
def upload(rows: list[dict], audio: dict):
    root=Path('/lab/data/paired-real-controls')
    (root/'wav').mkdir(parents=True,exist_ok=True)
    for name,data in audio.items(): (root/'wav'/name).write_bytes(data)
    manifest=json.loads(Path('/lab/data/realuser-v1/manifest-v1.json').read_text())
    for entry in manifest['sessions']:
        if entry['id'] not in ('realuser-v1-sob-028','realuser-v1-sob-032'): continue
        session=entry['session']
        name=entry['id']+'.wav'
        (root/'wav'/name).write_bytes((Path('/lab/data/realuser-v1')/entry['audio']['phone']).read_bytes())
        ayahs=list(dict.fromkeys(f"{x['from']['surah']}:{x['from']['ayah']}" for x in session['spans']))
        rows.append(dict(id=entry['id'],speaker=entry['speaker'],kind='omitted_word',ayahs=ayahs,wav='wav/'+name,scored=True,split='eval',events=session['events'],non_errors=[]))
    (root/'labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    vol.commit()
@app.local_entrypoint()
def main(listeners: str='spliced-clean-int8,paired-consonants-int8,paired-control-int8', main_checkout: str='../tilawa'):
    base=Path(main_checkout).resolve()/'.scratch/help'
    targets={'muo5gkq2_857a61f4':('omitted_word',94,5,2), 'muodfjiw_c16401bc':('omitted_word',26,106,4), 'muodhmzj_0639dae3':('wrong_word',79,39,2)}
    rows=[];audio={}
    for line in (base/'eval-labels.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['id'] not in targets:continue
        cls,s,a,w=targets[r['id']]
        r['events']=[dict(errorClass=cls,state='confirmed',location=dict(surah=s,ayah=a,word=w))]
        r['non_errors']=[];r['scored']=True;r['split']='eval'
        rows.append(r);audio[Path(r['wav']).name]=(base/r['wav']).read_bytes()
    assert len(rows)==3
    upload.remote(rows,audio)
    src,sha=lab_run.source()
    for name in listeners.split(','):
        cmd=f'pnpm --filter @tilawa/core build && pnpm --filter @tilawa/sdk exec tsx src/bench/real-controls-cli.ts --model /lab/data/models/{name} --help /lab/data/paired-real-controls --out "$LAB_OUT/report.json"'
        run=f'paired-real-controls-{name}-{sha}'
        result=lab_run.shard.remote(src,cmd,run,0,1)
        print(json.dumps(dict(run=run,**result)),flush=True)
