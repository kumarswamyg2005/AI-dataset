"""
MULTI-GENERATOR training (leave-one-generator-out) on genuine content-matched real-vs-AI.
Reuses cached OPF-15 / Base-63 features (no re-extraction).
Train real=nature + AI=ai pooled from ALL generators except the held-out one; test on held-out.
Compare to single-gen (SD v1.5) training baseline.
"""
import os, numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
SC = "/private/tmp/claude-501/-Users-kumaraswamy-Desktop-jpg-research/eb427cd5-6ebc-464d-a807-d7deeec6f45e/scratchpad/cache_v2"
def L(fn): z = np.load(f"{SC}/{fn}.npz"); return z["O"], z["B"]

# assemble per-family nature/ai feature pools
FAM = {"SD v1.5":"imagenet_ai_0424_sdv5","ADM":"imagenet_ai_0508_adm","GLIDE":"imagenet_glide",
       "Midjourney":"imagenet_midjourney","Wukong":"imagenet_ai_0424_wukong","VQDM":"imagenet_ai_0419_vqdm"}
NAT, AI = {}, {}
trR=L("tr_nature"); teR=L("te_nature_indom"); trA=L("tr_ai"); teA=L("te_ai_indom")
NAT["SD v1.5"] = (np.vstack([trR[0],teR[0]]), np.vstack([trR[1],teR[1]]))
AI ["SD v1.5"] = (np.vstack([trA[0],teA[0]]), np.vstack([trA[1],teA[1]]))
for name,d in FAM.items():
    if name=="SD v1.5": continue
    NAT[name]=L(f"re_{d}"); AI[name]=L(f"ai_{d}")

CAP=600  # per-generator contribution to training (keep balanced across makers)
def mk(R,A):
    X=np.vstack([R,A]); y=np.r_[np.zeros(len(R)),np.ones(len(A))]
    c=lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,max_depth=5,num_leaves=31,
        min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.1,
        class_weight="balanced",random_state=42,verbose=-1); c.fit(X,y); return c
def boot(clf,R,A,n=2000,seed=0):
    X=np.vstack([R,A]); y=np.r_[np.zeros(len(R)),np.ones(len(A))]; s=clf.predict_proba(X)[:,1]
    a=roc_auc_score(y,s); rng=np.random.default_rng(seed); m=len(y); arr=np.empty(n)
    for i in range(n):
        idx=rng.integers(0,m,m); yi=y[idx]; arr[i]=np.nan if yi.min()==yi.max() else roc_auc_score(yi,s[idx])
    lo,hi=np.nanpercentile(arr,[2.5,97.5]); return a,lo,hi

fams=list(FAM)
out=[f"MULTI-GENERATOR (leave-one-out) | per-maker cap {CAP} | boot 2000",
     f"{'held-out test':13s} {'OPF-15 [CI]':>21s} {'Base-63 [CI]':>21s} {'Comb-78 [CI]':>21s}  n(re/ai)"]
oo,bb,cc=[],[],[]
for H in fams:
    # train pool = all other families (col 0 = OPF, col 1 = Base)
    trRo=np.vstack([NAT[f][0][:CAP] for f in fams if f!=H]); trRb=np.vstack([NAT[f][1][:CAP] for f in fams if f!=H])
    trAo=np.vstack([AI [f][0][:CAP] for f in fams if f!=H]); trAb=np.vstack([AI [f][1][:CAP] for f in fams if f!=H])
    opf=mk(trRo,trAo); bas=mk(trRb,trAb); comb=mk(np.hstack([trRo,trRb]),np.hstack([trAo,trAb]))
    reO,reB=NAT[H]; aiO,aiB=AI[H]
    ao=boot(opf,reO,aiO); ab=boot(bas,reB,aiB); ac=boot(comb,np.hstack([reO,reB]),np.hstack([aiO,aiB]))
    out.append(f"{H:13s} {ao[0]:.3f}[{ao[1]:.3f},{ao[2]:.3f}] {ab[0]:.3f}[{ab[1]:.3f},{ab[2]:.3f}] "
               f"{ac[0]:.3f}[{ac[1]:.3f},{ac[2]:.3f}]  {len(reO)}/{len(aiO)}")
    oo.append(ao[0]); bb.append(ab[0]); cc.append(ac[0])
out.append(f"{'MEAN':13s} {np.mean(oo):.3f}{'':16s} {np.mean(bb):.3f}{'':16s} {np.mean(cc):.3f}")
out.append("")
out.append("vs SINGLE-gen (train SD v1.5 only): OPF 0.703  Base 0.745  Comb 0.807  (cross-mean)")
txt="\n".join(out); print(txt)
open(f"{SC}/../multigen_results.txt","w").write(txt+"\n")
