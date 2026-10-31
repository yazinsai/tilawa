"""Frozen practice-only decoding and scoring; no exam access."""
import argparse,json,re,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import lab
import lab_ledger as L
p=argparse.ArgumentParser();p.add_argument('arm',choices=['clean','phone']);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'.scratch/lab/experiments'
base=json.loads((out/'continuation-reference-main.json').read_text())
ids=[r['id'] for r in base['results'] if r['split']=='practice']
assert len(ids)==100 and len(set(ids))==100 and all(re.fullmatch('[a-zA-Z0-9.-]+',s) for s in ids)
listener='continuation-'+a.arm+'-int8'
subprocess.run(['modal','run','modal/lab_decode.py','--listener',listener,'--model','models/'+listener,'--test-set','lab-v1.5','--only',','.join(ids)],cwd=root,check=True)
args=SimpleNamespace(ledger='/ledger.json',main='main');led=lab.load(args);ts=L.test_set(led)
assert ts['id']=='lab-v1.5' and set(ids)==set(ts['sessions']['practice'])
if a.arm=='clean':
 subprocess.run([sys.executable,'tools/lab.py','run',str(root)],cwd=root,check=True)
 led=lab.load(args);result=L.latest(led['experiments']['listener-continuation-control'],'lab-v1.5')
 report=lab.vol_get('/runs/'+result['run']+'/score.json')
else:
 report=lab.score(args,root,name='continuation-phone-score',listener=listener,ts=ts,splits=('practice',),log=out/'continuation-phone-score.log')
assert len(report['results'])==100 and all(r['split']=='practice' for r in report['results'])
(out/('continuation-'+a.arm+'-score.json')).write_text(json.dumps(report,ensure_ascii=False,indent=1))
print(json.dumps(dict(arm=a.arm,run=report['run'],listener=listener)))
