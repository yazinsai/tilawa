"""Decode only the frozen practice IDs; score a companion arm without touching the exam."""
import argparse,json,re,subprocess,sys
from pathlib import Path
import lab
import lab_ledger as L
p=argparse.ArgumentParser();p.add_argument('listener',choices=['sparse-consonants-int8','sparse-random-int8']);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
base=json.loads(Path('/Users/rock/ai/projects/tilawa/.scratch/lab/cache/main-lab-v1.5-practice.score.json').read_text())
ids=[r['id'] for r in base['results'] if r['split']=='practice']
assert len(ids)==100 and len(set(ids))==100 and all(re.fullmatch('[a-zA-Z0-9.-]+',s) for s in ids)
subprocess.run(['modal','run','modal/lab_decode.py','--listener',a.listener,'--model','models/'+a.listener,'--test-set','lab-v1.5','--only',','.join(ids)],cwd=root,check=True)
if a.listener=='sparse-consonants-int8':
 subprocess.run([sys.executable,'tools/lab.py','run',str(root)],cwd=root,check=True)
else:
 from types import SimpleNamespace
 args=SimpleNamespace(ledger='/ledger.json',main='main')
 led=lab.load(args);ts=L.test_set(led)
 result=lab.score(args,root,name='sparse-random-score',listener=a.listener,ts=ts,splits=('practice',),log=Path('/tmp/sparse-random-score.log'))
 (root/'.scratch/lab/experiments/sparse-random-score.json').write_text(json.dumps(result,ensure_ascii=False,indent=1))
 print(json.dumps({'run':result['run'],'listener':result['listener'],'testSet':result['testSet']}))
