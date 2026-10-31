"""Freeze equal-size arms after independent lexical as-said verification, before training."""
import argparse,hashlib,json,random,re,unicodedata
from pathlib import Path

def normalized(text):
    text=''.join(c for c in text if unicodedata.category(c) not in ('Mn','Me') and c!='ـ')
    text=text.translate(str.maketrans({'ٱ':'ا','أ':'ا','إ':'ا','آ':'ا'}))
    return re.findall(r'[\u0621-\u064a]+',text)

def lexical_match(words,transcript):
    from review_asr_core import plain_spellings, norm_variants
    heard=normalized(transcript)
    if len(heard)!=len(words):return False
    for word,h in zip(words,heard):
        variants={tuple(normalized(p)) for p in (plain_spellings(word['text']) + list(norm_variants(word['text'])) + [word['text']])}
        if (h,) not in variants:return False
    return True


def build(mining,asr,controls):
    proof={}; valid={}
    by={r['key']:r for r in mining['rows']}
    for key in set(mining['arms']['selected']+mining['arms']['random']):
        r=by[key]; heard=asr.get(key,{})
        expected=' '.join(w['text'] for w in r['words'])
        match=not heard.get('error') and lexical_match(r['words'],heard.get('transcript',''))
        proof[key]={'audioSha256':r['audioSha256'],'independentAsSaidMatch':match,
                    'expected':expected,'heard':heard.get('transcript'),
                    'reviewRecord':'/reviews/sparse-training-audit/asr-soniox.json#'+key,
                    'basis':'Source label, teacher and clean student agree at phoneme level; independent blind Soniox lexical transcript matches. No fine pronunciation judgment.'}
    def accepted(arm):return [k for k in mining['arms'][arm] if proof[k]['independentAsSaidMatch']]
    n=min(len(accepted('selected')),len(accepted('random')))
    if not n:raise ValueError('No independently verified matched subset; no training')
    for j,arm in enumerate(('selected','random')):
        pool=accepted(arm);random.Random(810+j).shuffle(pool);valid[arm]=sorted(pool[:n])
    return {**valid,'poolSha256':hashlib.sha256('\n'.join(mining['poolKeys']).encode()).hexdigest(),
        'policy':{'ctcWeight':1.,'kdWeight':2.,'updates':200,'weight':3.},
        'verifiedTargets':proof,'selfCorrectionControls':controls,
        'verificationCounts':{'selected':len(accepted('selected')),'random':len(accepted('random')),'pairedSize':n},
        'sourceMining':'/student/runs/sparse-mining-v1/mining.json'}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mining',required=True);p.add_argument('--asr',required=True);p.add_argument('--controls',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 d=build(json.loads(Path(a.mining).read_text()),json.loads(Path(a.asr).read_text()),json.loads(Path(a.controls).read_text()))
 Path(a.out).write_text(json.dumps(d,ensure_ascii=False,indent=1));print(d['verificationCounts'])
