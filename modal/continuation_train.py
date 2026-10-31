"""Frozen uniform paired control: only phone_p differs between arms."""
import json, subprocess
from pathlib import Path
import modal
import student_train as S
app=modal.App('tilawa-continuation-control',image=S.image.add_local_python_source('student_train'))
app.include(S.app)
@app.local_entrypoint()
def main(arm: str):
    assert arm in ('clean','phone')
    cfg=dict(name='continuation-'+arm+'-v2',size='full',max_hours=50.,max_steps=200,epochs=1,eval_max=300,max_batch_s=800.,base_lr=.0015,target='spliced',init_run='spliced-clean',phone_p=0. if arm=='clean' else .75,ctc_weight=1.,kd_weight=2.,seed=0,deterministic_aug=True,expected_pool_sha256='ba66b2fb1c88ebf89fe260805cdc351f33bbb7347f577ef74ed819e06c057131',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    Path('.scratch/lab/experiments/continuation-'+arm+'-config.json').write_text(json.dumps(cfg,indent=1))
    print(json.dumps(S.train.remote(cfg),ensure_ascii=False),flush=True)
