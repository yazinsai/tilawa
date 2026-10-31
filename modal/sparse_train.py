"""Exactly two matched sparse arms, with mandatory frozen verification manifest."""
import modal
import json
import subprocess
from pathlib import Path
import student_train as S
app=modal.App('tilawa-sparse-train',image=S.image.add_local_python_source('student_train'))
app.include(S.app)
@app.local_entrypoint()
def main(arm: str):
 if arm not in ('selected','random'):raise ValueError('arm must be selected or random')
 # Fixed-budget original single-view objective; never activate paired consistency.
 init_run=json.loads(Path('.scratch/lab/experiments/sparse-objective-audit.json').read_text())['initRun']
 source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 print(S.train.remote(dict(name='sparse-consonants-v1' if arm=='selected' else 'sparse-random-v1',
   size='full',max_hours=50.,max_steps=200,epochs=6,max_batch_s=800.,base_lr=.0015,
   target='spliced',init_run=init_run,phone_p=.75,ctc_weight=1.,kd_weight=2.,seed=0,
   sparse_manifest='/student/runs/sparse-mining-v1/verified-manifest.json',sparse_arm=arm,source_sha=source_sha)))
