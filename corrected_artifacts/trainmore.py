"""
TRAIN-MORE: full-data multi-generator leave-one-out on genuine content-matched real-vs-AI.
Extract up to 1500/class per generator (was 600). Per-set caching -> resumable.
Leave-one-generator-out (test maker NEVER in training). OPF-15 / Base-63 / Combined, boot CIs.
"""
import sys, os, glob, time, numpy as np, cv2
from io import BytesIO
from PIL import Image
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, "/Users/kumaraswamy/Desktop/jpg/research/full")
import opf_local_train as L
SC = "/private/tmp/claude-501/-Users-kumaraswamy-Desktop-jpg-research/eb427cd5-6ebc-464d-a807-d7deeec6f45e/scratchpad"
CDIR = f"{SC}/cache_more"; os.makedirs(CDIR, exist_ok=True)
ROOT = "/Users/kumaraswamy/Desktop/jpg/research/data/tiny_genimage"
S, Q, SUB, NPC = 256, 95, 0, 1500     # up to 1500 kept per class per generator
FAM = {"SD v1.5":"imagenet_ai_0424_sdv5","ADM":"imagenet_ai_0508_adm","GLIDE":"imagenet_glide",
       "Midjourney":"imagenet_midjourney","Wukong":"imagenet_ai_0424_wukong","VQDM":"imagenet_ai_0419_vqdm"}

def norm(path):
    bgr = cv2.imread(str(path))
    if bgr is None: return None
    H, W = bgr.shape[:2]
    if H < S or W < S: return None
    t, l = (H-S)//2, (W-S)//2
    im = Image.fromarray(cv2.cvtColor(bgr[t:t+S, l:l+S], cv2.COLOR_BGR2RGB))
    buf = BytesIO(); im.save(buf, format="JPEG", quality=Q, subsampling=SUB); buf.seek(0)
    return cv2.cvtColor(np.array(Image.open(buf).convert("RGB")), cv2.COLOR_RGB2BGR)

def base_feat(bgr):
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    H, W = y.shape; nh, nw = H//8, W//8
    acc = np.empty((nh*nw, 64), np.float32); b = 0
    for r in range(nh):
        for c in range(nw): acc[b] = cv2.dct(y[r*8:r*8+8, c*8:c*8+8]).flatten(); b += 1
    return np.log1p(acc.var(axis=0)[1:]).astype(np.float32)

def extract(paths, tag):
    O, B = [], []; t0=time.time()
    for p in paths:
        if len(O) >= NPC: break
        bgr = norm(p)
        if bgr is None: continue
        of = L._extract_from_bgr(bgr); bs = base_feat(bgr)
        if of is not None and bs is not None and np.all(np.isfinite(of)) and np.all(np.isfinite(bs)):
            O.append(of); B.append(bs)
    print(f"  [{tag}] {len(O)} in {time.time()-t0:.0f}s", flush=True)
    return np.array(O, np.float32), np.array(B, np.float32)

def gf(fam, cls): return sorted(glob.glob(f"{ROOT}/{FAM[fam]}/train/{cls}/*"))
def cext(fam, cls):
    fp = f"{CDIR}/{FAM[fam]}_{cls}.npz"
    if os.path.exists(fp):
        z = np.load(fp); print(f"  [{fam} {cls}] cached {len(z['O'])}", flush=True); return z["O"], z["B"]
    O, B = extract(gf(fam, cls)[:NPC+400], f"{fam} {cls}")   # feed extra to offset crop drops
    np.savez(fp, O=O, B=B); return O, B

NAT, AI = {}, {}
for fam in FAM:
    NAT[fam] = cext(fam, "nature"); AI[fam] = cext(fam, "ai")

# leave-one-generator-out (test maker never trained)
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
    return a, *np.nanpercentile(arr,[2.5,97.5])

fams=list(FAM); out=[f"TRAIN-MORE full-data leave-one-out (up to {NPC}/class/maker) | boot 2000",
    f"{'held-out':13s} {'OPF-15':>18s} {'Base-63':>18s} {'Comb-78':>18s}  train_n"]
oo,bb,cc=[],[],[]
for H in fams:
    trRo=np.vstack([NAT[f][0] for f in fams if f!=H]); trRb=np.vstack([NAT[f][1] for f in fams if f!=H])
    trAo=np.vstack([AI[f][0]  for f in fams if f!=H]); trAb=np.vstack([AI[f][1]  for f in fams if f!=H])
    opf=mk(trRo,trAo); bas=mk(trRb,trAb); comb=mk(np.hstack([trRo,trRb]),np.hstack([trAo,trAb]))
    reO,reB=NAT[H]; aiO,aiB=AI[H]
    ao=boot(opf,reO,aiO); ab=boot(bas,reB,aiB); ac=boot(comb,np.hstack([reO,reB]),np.hstack([aiO,aiB]))
    out.append(f"{H:13s} {ao[0]:.3f}[{ao[1]:.3f},{ao[2]:.3f}] {ab[0]:.3f}[{ab[1]:.3f},{ab[2]:.3f}] "
               f"{ac[0]:.3f}[{ac[1]:.3f},{ac[2]:.3f}]  {len(trRo)}+{len(trAo)}")
    if H!="SD v1.5": oo.append(ao[0]); bb.append(ab[0]); cc.append(ac[0])
out.append(f"{'mean(5 unseen)':13s} {np.mean(oo):.3f}{'':13s} {np.mean(bb):.3f}{'':13s} {np.mean(cc):.3f}")
out.append("")
out.append("prev (cap-600 multi-gen): OPF 0.719  Base 0.891  Comb 0.895  (5-unseen mean)")
out.append("deep zero-shot          : NPR 0.630  FatFormer 0.844")
txt="\n".join(out); print("\n"+txt, flush=True)
open(f"{SC}/trainmore_results.txt","w").write(txt+"\n"); print("WROTE trainmore_results.txt", flush=True)
