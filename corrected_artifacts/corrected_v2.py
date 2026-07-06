"""
CORRECTED in-domain + cross-gen experiment on GENUINE, content-matched real-vs-AI data.
Replaces the paper's mislabeled Tables (where in-domain 'AI' was Shutterstock real photos).

Data:   tiny_genimage/imagenet_*/train/{nature,ai}  (real ImageNet vs genuine AI, same classes)
Norm:   native center crop 256 + re-encode JPEG Q95 4:4:4, applied identically to both classes
        (the paper's fair, no-resize bias-controlled protocol).
Train:  SD v1.5 nature(real) vs ai(AI).
Test:   held-out SD v1.5 (in-domain) + 5 unseen generators, each with its OWN nature (content-matched,
        no cross-family leakage: nature filenames are unique per family).
Report: AUC (AI = positive) with bootstrap 95% CI (2000 resamples) for OPF-15, Base-63, Combined-78,
        plus in-domain accuracy@0.5. BigGAN excluded (128px < 256 crop), as in the paper.
Env:    SMOKE=1 -> tiny counts for a fast correctness check.
"""
import sys, os, glob, time, numpy as np, cv2
from io import BytesIO
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
sys.path.insert(0, "/Users/kumaraswamy/Desktop/jpg/research/full")
import opf_local_train as L

SMOKE = os.environ.get("SMOKE", "0") == "1"
S, Q, SUB = 256, 95, 0
ROOT = "/Users/kumaraswamy/Desktop/jpg/research/data/tiny_genimage"
SCRATCH = "/private/tmp/claude-501/-Users-kumaraswamy-Desktop-jpg-research/eb427cd5-6ebc-464d-a807-d7deeec6f45e/scratchpad"
FAM = {"SD v1.5 (in-dom)":"imagenet_ai_0424_sdv5","ADM":"imagenet_ai_0508_adm",
       "GLIDE":"imagenet_glide","Midjourney":"imagenet_midjourney",
       "Wukong":"imagenet_ai_0424_wukong","VQDM":"imagenet_ai_0419_vqdm"}
INDOM = "SD v1.5 (in-dom)"
# counts: feed extra nature to offset ~10% dropped by the 256 crop
if SMOKE:
    N_TR_NAT, N_TR_AI, TE_NAT, TE_AI, XN_NAT, XN_AI, NBOOT = 80, 70, 60, 50, 60, 50, 200
    CACHE = f"{SCRATCH}/tinygen_v2_smoke.npz"
else:
    N_TR_NAT, N_TR_AI, TE_NAT, TE_AI, XN_NAT, XN_AI, NBOOT = 1500, 1500, 500, 500, 660, 600, 2000
    CACHE = f"{SCRATCH}/tinygen_v2.npz"

def norm(path):
    bgr = cv2.imread(str(path))
    if bgr is None: return None
    H, W = bgr.shape[:2]
    if H < S or W < S: return None
    t, l = (H-S)//2, (W-S)//2
    crop = bgr[t:t+S, l:l+S]
    im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    buf = BytesIO(); im.save(buf, format="JPEG", quality=Q, subsampling=SUB); buf.seek(0)
    return cv2.cvtColor(np.array(Image.open(buf).convert("RGB")), cv2.COLOR_RGB2BGR)

def base_feat(bgr):
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    H, W = y.shape; nh, nw = H//8, W//8
    if nh*nw < 4: return None
    acc = np.empty((nh*nw, 64), np.float32); b = 0
    for r in range(nh):
        for c in range(nw): acc[b] = cv2.dct(y[r*8:r*8+8, c*8:c*8+8]).flatten(); b += 1
    return np.log1p(acc.var(axis=0)[1:]).astype(np.float32)

def extract(paths, tag):
    O, B = [], []; t0 = time.time()
    for p in paths:
        bgr = norm(p)
        if bgr is None: continue
        of = L._extract_from_bgr(bgr); bs = base_feat(bgr)
        if of is not None and bs is not None and np.all(np.isfinite(of)) and np.all(np.isfinite(bs)):
            O.append(of); B.append(bs)
    print(f"  [{tag}] kept {len(O)}/{len(paths)} in {time.time()-t0:.0f}s", flush=True)
    return np.array(O, np.float32), np.array(B, np.float32)

def gf(fam, cls): return sorted(glob.glob(f"{ROOT}/{FAM[fam]}/train/{cls}/*"))

# ---- feature extraction with PER-SET caching (resumable across foreground calls) ----
CDIR = f"{SCRATCH}/cache_v2{'_smoke' if SMOKE else ''}"; os.makedirs(CDIR, exist_ok=True)
def cached_extract(paths, key, tag):
    fp = f"{CDIR}/{key}.npz"
    if os.path.exists(fp):
        z = np.load(fp); print(f"  [{tag}] cached {len(z['O'])}", flush=True); return z["O"], z["B"]
    O, B = extract(paths, tag); np.savez(fp, O=O, B=B); return O, B

nat, ai = gf(INDOM, "nature"), gf(INDOM, "ai")
# define every set: (Dkey, cachekey, paths, tag)
SETS = [
    ("trR", "tr_nature", nat[:N_TR_NAT], "sdv5 train nature"),
    ("trA", "tr_ai",     ai[:N_TR_AI],   "sdv5 train ai"),
    (("re", INDOM), "te_nature_indom", nat[N_TR_NAT:N_TR_NAT+TE_NAT], "sdv5 test nature"),
    (("ai", INDOM), "te_ai_indom",     ai[N_TR_AI:N_TR_AI+TE_AI],     "sdv5 test ai"),
]
for fam in FAM:
    if fam == INDOM: continue
    SETS.append(((("re", fam)), f"re_{FAM[fam]}", gf(fam, "nature")[:XN_NAT], f"{fam} nature"))
    SETS.append(((("ai", fam)), f"ai_{FAM[fam]}", gf(fam, "ai")[:XN_AI],     f"{fam} ai"))
D = {}
for dkey, ckey, paths, tag in SETS:
    D[dkey] = cached_extract(paths, ckey, tag)
print("all sets ready", flush=True)

# ---- train (same classifier/hyperparams as the paper) ----
def mk(R, A):
    X = np.vstack([R, A]); y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    c = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        class_weight="balanced", random_state=42, verbose=-1); c.fit(X, y); return c

def boot(clf, R, A, nboot=NBOOT, seed=0):
    X = np.vstack([R, A]); y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    s = clf.predict_proba(X)[:, 1]; a = roc_auc_score(y, s)
    rng = np.random.default_rng(seed); m = len(y); arr = np.empty(nboot)
    for i in range(nboot):
        idx = rng.integers(0, m, m); yi = y[idx]
        arr[i] = np.nan if yi.min()==yi.max() else roc_auc_score(yi, s[idx])
    lo, hi = np.nanpercentile(arr, [2.5, 97.5]); return a, lo, hi

trR_O, trR_B = D["trR"]; trA_O, trA_B = D["trA"]
opf = mk(trR_O, trA_O); bas = mk(trR_B, trA_B)
comb = mk(np.hstack([trR_O, trR_B]), np.hstack([trA_O, trA_B]))

# ---- evaluate ----
hdr = f"CORRECTED (genuine content-matched real-vs-AI | native {S} Q{Q} 4:4:4 | boot={NBOOT})"
L1 = [hdr, f"train: SD v1.5  nature(real) {len(trR_O)}  vs  ai(AI) {len(trA_O)}",
      f"{'test set':17s} {'OPF-15 [95% CI]':>22s} {'Base-63 [95% CI]':>22s} {'Comb-78 [95% CI]':>22s}  n(re/ai)"]
opfs, bass, combs = [], [], []
for fam in FAM:
    reO, reB = D[("re", fam)]; aiO, aiB = D[("ai", fam)]
    ao, lo1, hi1 = boot(opf, reO, aiO)
    ab, lo2, hi2 = boot(bas, reB, aiB)
    ac, lo3, hi3 = boot(comb, np.hstack([reO,reB]), np.hstack([aiO,aiB]))
    L1.append(f"{fam:17s} {ao:.3f}[{lo1:.3f},{hi1:.3f}] {ab:.3f}[{lo2:.3f},{hi2:.3f}] "
              f"{ac:.3f}[{lo3:.3f},{hi3:.3f}]  {len(reO)}/{len(aiO)}")
    if fam != INDOM: opfs.append(ao); bass.append(ab); combs.append(ac)
L1.append(f"{'mean(5 unseen)':17s} {np.mean(opfs):.3f}{'':17s} {np.mean(bass):.3f}{'':17s} {np.mean(combs):.3f}")
# in-domain accuracy@0.5
reO, reB = D[("re", INDOM)]; aiO, aiB = D[("ai", INDOM)]
def acc(clf, R, A):
    X = np.vstack([R, A]); y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    return accuracy_score(y, (clf.predict_proba(X)[:,1] >= 0.5).astype(int))
L1.append(f"in-domain accuracy@0.5:  OPF {acc(opf,reO,aiO):.3f}   Base {acc(bas,reB,aiB):.3f}   "
          f"Comb {acc(comb,np.hstack([reO,reB]),np.hstack([aiO,aiB])):.3f}")
out = "\n".join(L1); print("\n"+out, flush=True)
open(f"{SCRATCH}/corrected_v2_results.txt","w").write(out+"\n")
print("\nWROTE corrected_v2_results.txt", flush=True)
