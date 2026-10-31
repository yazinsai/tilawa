"""Read-only aligned wider-window diagnostic; no decoder changes."""
import collections, json, os
from pathlib import Path
import lab_phonemes_core as P
D=Path('/lab/data/lab-v1.5/sessions')
original=json.loads(Path('/app/tools/blank-original-diagnostic.json').read_text())
tokens=json.loads(Path('/app/packages/core/assets/ref-v2.json').read_text())['inventory']['tokens']
cons=set('ءبتثجحخدذرزسشصضطظعغفقكلمنهوي')
rows=[]
for frozen in original:
 x=frozen['incident']
 s=json.loads((D/f"{x['session']}.json").read_text())
 cols=frozen['columns']
 focus=[(i,c) for i,c in enumerate(cols)]
 if not focus: continue
 for variant in ('phone','studio'):
  ev=json.loads((D/f"{x['session']}.spliced-clean-int8.{variant}.evidence.json").read_text())
  for ci,c in focus:
   a,b=c['startS'],c['endS']; narrow=P.align([(p,0) for p in c['expected']],P.heard_in(ev,a-.12,b+.12),[(a,b)])
   for p in sorted({o['e'] for o in frozen[variant]['ops'] if o['op']=='del' and o.get('e') in cons and o.get('col')==ci}):
    record={'incident':x,'variant':variant,'token':p,'wordSpan':[a,b],'windows':[]}
    for margin in (0,.12,.24,.40):
     # Include neighboring words so their emissions cannot be borrowed for free.
     wide=P.window_columns(s,None,a-margin-.01,b+margin+.01)
     wi=next(i for i,w in enumerate(wide) if w['at']==c['at'] and w['startS']==a)
     ops=P.align([(q,i) for i,w in enumerate(wide) for q in w['expected']],P.heard_in(ev,a-margin,b+margin),[(w['startS'],w['endS']) for w in wide],tol=margin)
     tid=tokens.index(p)+1; peaks=[]
     for f in range(max(0,int((a-margin)*1000/ev['frameMs'])),min(ev['nFrames'],int((b+margin)*1000/ev['frameMs'])+1)):
      post=ev['posteriors']; k=post['k']; ids=post['ids'][f*k:(f+1)*k]; probs=post['probs'][f*k:(f+1)*k]
      if tid in ids: peaks.append((probs[ids.index(tid)],f*ev['frameMs']/1000,ids[0]==tid))
     peak=max(peaks,default=(0,None,False))
     record['windows'].append({'margin':margin,'maxP':peak[0],'peakS':peak[1],'greedyAtPeak':peak[2],'alignedOps':[o for o in ops if o.get('col')==wi and o.get('e')==p]})
    rows.append(record)
summary={v:{'pairs':len(rr:=[r for r in rows if r['variant']==v]),'weakNarrow':sum(r['windows'][0]['maxP']<.3 for r in rr),'weakWide':sum(r['windows'][-1]['maxP']<.3 for r in rr),'newWidePeaks':sum(r['windows'][0]['maxP']<.3<=r['windows'][-1]['maxP'] for r in rr)} for v in ('phone','studio')}
out=Path(os.environ['LAB_OUT']); (out/'wide-evidence.json').write_text(json.dumps({'summary':summary,'rows':rows},ensure_ascii=False,indent=1)); print(json.dumps(summary))
