"""Recreate source word boundaries from the frozen local tlog training labels."""
import argparse,gzip,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',default='tools/sparse-word-metadata.json.gz');a=p.parse_args()
ref=json.loads(Path('packages/core/assets/ref-v2.json').read_text());words=[]
for ay in ref['ayahs']:
 for j,w in enumerate(ay['w']):
  words.append({'at':{'surah':ay['s'],'ayah':ay['a'],'word':j+1},'text':w,'phones':ay['p'][ay['o'][j]:ay['o'][j+1] if j+1<len(ay['o']) else len(ay['p'])]})
ix={(w['at']['surah'],w['at']['ayah'],w['at']['word']):i for i,w in enumerate(words)}
out={};src=Path(a.source)
for line in src.read_text().splitlines():
 c=json.loads(line)
 if c['label']['kind']!='span':continue
 start,end=c['label']['span']['start'],c['label']['span']['end']
 lo=ix[tuple(start[k] for k in ('surah','ayah','word'))];hi=ix[tuple(end[k] for k in ('surah','ayah','word'))]
 out[f"{c['shard']}:{c['row']}"]={'words':words[lo:hi+1],'sourcePer':c['per'],'sourceConfidence':c['confidence']}
with gzip.open(a.out,'wt') as f:json.dump(out,f,ensure_ascii=False,separators=(',',':'))
Path(a.out+'.provenance.json').write_text(json.dumps({'sourceSha256':hashlib.sha256(src.read_bytes()).hexdigest(),'rows':len(out)},indent=1))
