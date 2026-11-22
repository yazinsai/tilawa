"""Group failure incidents and compare teacher evidence; automated claims remain provisional."""
import argparse,json,collections
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--student',required=True);ap.add_argument('--teacher',required=True);ap.add_argument('--findings',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 student=json.loads(Path(a.student).read_text());teacher={c['id']:c for c in json.loads(Path(a.teacher).read_text())['clips']};findings=json.loads(Path(a.findings).read_text())['findings'];reviewed=collections.defaultdict(list)
 for f in findings:reviewed[f['clip']].append(f)
 incidents={}
 for c in student['clips']:
  arm=c['arms']['leadFree'];t=teacher.get(c['id']);ta=t['arms'].get('leadFree',{}) if t else {}
  for kind,records in [('miss',arm.get('misses',[])),('unmatched_warning',arm.get('falsePositives',[]))]:
   for record in records:
    at=record['at'] if isinstance(record,dict) else record.split('/',1)[1]
    key=c['id']+'/'+at
    incident=incidents.setdefault(key,{'clip':c['id'],'at':at,'records':[],'humanFindings':reviewed[c['id']],'audio':c['audio'],'words':[w for w in c['words'] if w['at']==at],'labels':c['labels'],'teacherAvailable':t is not None})
    incident['records'].append({'kind':kind,'record':record})
    incident['teacherEvents']=[e for e in ta.get('engineEvents',[]) if e.get('at')==at]
    incident['teacherMisses']=ta.get('misses',[])
 for i in incidents.values():
  labels=[l for l in i['labels'] if ':'.join(str(l['location'][k]) for k in ['surah','ayah','word'])==i['at']]
  i['hypotheses']=[]
  for r in i['records']:
   if r['kind']=='unmatched_warning':
    w=r['record'];adj=[l for l in i['labels'] if l['location']['surah']==int(i['at'].split(':')[0]) and l['location']['ayah']==int(i['at'].split(':')[1]) and abs(l['location']['word']-int(i['at'].split(':')[2]))==1]
    if labels:i['hypotheses'].append('same-word label/report diagnosis or state mismatch')
    elif adj:i['hypotheses'].append('adjacent labelled error: inspect warning span and location before calling a false alarm')
    else:i['hypotheses'].append('unlabelled warning: listener disagreement or missing label requires audio verification')
  i['queue']='reviewed' if i['humanFindings'] else 'independent_audio_opinion'
  i['teacherAgreementIsIndependent']=False
 out={'version':1,'studentSource':a.student,'teacherSource':a.teacher,'arm':'leadFree','limitations':['Teacher/student agreement is correlated evidence, not human truth.','Clip-level human findings do not resolve unrelated incidents in the same recording.','Labels and benchmark metrics remain unchanged.'],'summary':dict(collections.Counter(i['queue'] for i in incidents.values())),'incidents':list(incidents.values())}
 Path(a.out).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'failureRecords':sum(len(i['records']) for i in incidents.values()),'incidents':len(incidents),'queues':out['summary']}))
if __name__=='__main__':main()
