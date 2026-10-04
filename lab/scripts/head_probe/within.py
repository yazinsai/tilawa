import numpy as np, sys
sys.argv=["x"]
exec(open("score.py").read().split('if __name__')[0])
from collections import defaultdict
dev=[x for x in rows if x["split"]=="dev" and not x["ign"]]
yd=np.array([x["y"] for x in dev]); gd=np.array([x["spk"] for x in dev])
for which,C in ((("rule","lp"),1.0),(("enc",),0.001),(("rule","lp","enc"),0.01)):
    Xd=np.stack([feats(x,which) for x in dev]); oof=np.zeros(len(dev))
    for tr,te in GroupKFold(5).split(Xd,yd,gd): oof[te]=model(C).fit(Xd[tr],yd[tr]).predict_proba(Xd[te])[:,1]
    # within acted clip: rank of label word among its clip's words
    byc=defaultdict(list)
    for x,s in zip(dev,oof):
        if x["set"]=="help-acted": byc[x["cid"]].append((s,x["y"]))
    ranks=[];top1=0;n=0
    for c,l in byc.items():
        if not any(y for _,y in l) or len(l)<3: continue
        sp=[s for s,y in l if y][0]; others=[s for s,y in l if not y]
        ranks.append(np.mean([sp>o for o in others])); top1+=sp>max(others); n+=1
    # clip-level: acted vs clean mean score
    acted=np.mean([s for x,s in zip(dev,oof) if x["set"]=="help-acted" and not x["y"]]); clean=np.mean([s for x,s in zip(dev,oof) if x["set"]=="help-clean"])
    print(which, "within-clip AUC %.3f top1 %d/%d | mean score acted-nonlabel %.3f clean %.3f"%(np.mean(ranks),top1,n,acted,clean))
