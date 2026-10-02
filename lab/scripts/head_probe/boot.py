import numpy as np, sys, json
sys.argv=["x"]
exec(open("score.py").read().split('if __name__')[0])
MAN={s["id"]:s for s in json.load(open("/tmp/help/manifest.json"))["samples"]}
def fit(which,C):
    dev=[x for x in rows if x["split"]=="dev" and not x["ign"]]; test=[x for x in rows if x["split"]=="test" and not x["ign"]]
    Xd=np.stack([feats(x,which) for x in dev]); yd=np.array([x["y"] for x in dev]); gd=np.array([x["spk"] for x in dev])
    oof=np.zeros(len(dev))
    for tr,te in GroupKFold(5).split(Xd,yd,gd): oof[te]=model(C).fit(Xd[tr],yd[tr]).predict_proba(Xd[te])[:,1]
    st=model(C).fit(Xd,yd).predict_proba(np.stack([feats(x,which) for x in test]))[:,1]
    cd=sorted((oof[i] for i,x in enumerate(dev) if x["set"]=="help-clean" and (x["ayah"],x["w"]) not in rule_flagged(x["cid"])),reverse=True)
    return test,st,cd
def flags(units,scores,thr):
    out=defaultdict(set)
    for x,s in zip(units,scores):
        if s>=thr: out[x["cid"]].add((x["ayah"],x["w"]))
    return out
def per_clip(hf):
    d={}
    for cid,(s,r) in BASE.items():
        if r["split"]!="test": continue
        fl={(i["ayah"],i["word"]) for i in r["issues"] if i["kind"] in ERR}|hf.get(cid,set())
        if s=="help-clean": d[cid]=(0,0,0,len(fl)); continue
        lab=LABELS[cid]; h=[f for f in fl if hit(f,lab)]
        d[cid]=(int(lab["label_kind"] in IN_SCOPE), int(lab["label_kind"] in IN_SCOPE and bool(h)), len(h), len(fl))
    return d
def f1(d,ids):
    a=np.array([d[i] for i in ids]); n,c,tp,fl=a.sum(0); P=tp/fl if fl else 0; R=c/n if n else 0
    return 2*P*R/(P+R) if P+R else 0, R
base=per_clip({})
spk=defaultdict(list)
for cid in base: spk[MAN[cid]["speaker"]].append(cid)
S=list(spk); rng=np.random.default_rng(0)
for which,C,ks in ((("rule","lp","enc"),0.01,(1,2,4)),(("enc",),0.001,(0,1))):
    test,st,cd=fit(which,C)
    for k in ks:
        cand=per_clip(flags(test,st,cd[k]+1e-9))
        dF=[];dR=[]
        for _ in range(2000):
            ids=[c for s in rng.choice(S,len(S)) for c in spk[s]]
            fa,ra=f1(cand,ids); fb,rb=f1(base,ids); dF.append(fa-fb); dR.append(ra-rb)
        fa,ra=f1(cand,list(base)); fb,rb=f1(base,list(base))
        print('+'.join(which),k,"dF1 %+.3f [%+.3f, %+.3f]  dR %+.3f [%+.3f, %+.3f]"%(fa-fb,*np.percentile(dF,[2.5,97.5]),ra-rb,*np.percentile(dR,[2.5,97.5])))
