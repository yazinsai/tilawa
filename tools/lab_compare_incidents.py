"""Practice incident comparison, preserving the prior listener matching policy.

Usage: python3 tools/lab_compare_incidents.py BASE.json NEW.json > comparison.json
Warnings match one-to-one by session/word within six seconds. A changed warning
class is reported separately. Inputs may include exam records; those are ignored.
"""
import json,sys,collections
b=json.load(open(sys.argv[1]));n=json.load(open(sys.argv[2]))
assert b['testSet']==n['testSet']
for report in (b,n):
 report['results']=[r for r in report['results'] if r['split']=='practice']
 report['cases']=[c for c in report['cases'] if c['split']=='practice']
assert b['results'] and n['results']
assert [(r['id'],r['speaker'],r['durationS'],r['recitedS']) for r in b['results']]==[(r['id'],r['speaker'],r['durationS'],r['recitedS']) for r in n['results']]
out={'baselineRun':b['run'],'newRun':n['run'],'testSet':b['testSet'],'baselineSha':b['sha'],'newSha':n['sha'],'warningMatching':'chronological one-to-one nearest same session/word within 6 seconds (historical listener policy)','variants':{}}
for v in ['phone','studio']:
 bc=[c for c in b['cases'] if c['variant']==v];nc=[c for c in n['cases'] if c['variant']==v]
 bm={c['key']:c for c in bc if c['kind']=='mistake'};nm={c['key']:c for c in nc if c['kind']=='mistake'}
 assert bm.keys()==nm.keys()
 for k in bm:
  assert all(bm[k][f]==nm[k][f] for f in ['session','errorClass','word','startS','endS'])
 recovered=[c for k,c in nm.items() if bm[k]['outcome']=='miss' and c['outcome']!='miss']
 missed=[c for k,c in nm.items() if bm[k]['outcome']!='miss' and c['outcome']=='miss']
 bw=[c for c in bc if c['kind']=='false_warning'];nw=[c for c in nc if c['kind']=='false_warning']
 matched=set();new=[];class_changes=[]
 for w in sorted(nw,key=lambda w:w['timeS']):
  choices=[(abs(a['timeS']-w['timeS']),i) for i,a in enumerate(bw) if i not in matched and a['session']==w['session'] and a['word']==w['word'] and abs(a['timeS']-w['timeS'])<=6]
  if choices:
   i=min(choices)[1];matched.add(i)
   if bw[i]['errorClass']!=w['errorClass']:class_changes.append({'base':bw[i],'new':w})
  else:new.append(w)
 miss_counts=[sum(c['outcome']=='miss' for c in bm.values()),sum(c['outcome']=='miss' for c in nm.values())]
 removed=[w for i,w in enumerate(bw) if i not in matched]
 assert miss_counts[1]-miss_counts[0]==len(missed)-len(recovered)
 assert len(nw)-len(bw)==len(new)-len(removed)
 for report,miss_count,warns in [(b,miss_counts[0],bw),(n,miss_counts[1],nw)]:
  assert sum(r['variants'][v]['missed'] for r in report['results'])==miss_count
  assert sum(r['variants'][v]['falseWarnings'] for r in report['results'])==len(warns)
 out['variants'][v]={'misses':miss_counts, 'warnings':[len(bw),len(nw)],'recoveries':recovered,'newMisses':missed,'newWarnings':new,'removedWarnings':removed,'warningClassChanges':class_changes, 'recoveriesByChapter':dict(collections.Counter(c['word'].split(':')[0] for c in recovered))}
print(json.dumps(out,ensure_ascii=False,indent=1))
