"""Eval-only rerun of the SD-Turbo modern-generator row.
Features are already cached by sdxl_row.py; this script avoids importing torch
(torch+lightgbm OpenMP clash segfaulted the combined script twice on macOS).
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

AD = "/Users/kumaraswamy/Desktop/jpg/research/full/corrected_artifacts"
TAG, MODEL = "sd_turbo", "stabilityai/sd-turbo"
FAMD = {"SD v1.5":"imagenet_ai_0424_sdv5","ADM":"imagenet_ai_0508_adm","GLIDE":"imagenet_glide",
        "Midjourney":"imagenet_midjourney","Wukong":"imagenet_ai_0424_wukong","VQDM":"imagenet_ai_0419_vqdm"}

z = np.load(f"{AD}/cache_{TAG}/mj_reals.npz"); reO, reB = z["O"], z["B"]
z = np.load(f"{AD}/cache_{TAG}/{TAG}_ai.npz"); aiO, aiB = z["O"], z["B"]

NAT, AI = {}, {}
for f, d in FAMD.items():
    zN = np.load(f"{AD}/cache_more/{d}_nature.npz"); zA = np.load(f"{AD}/cache_more/{d}_ai.npz")
    NAT[f] = (zN["O"], zN["B"]); AI[f] = (zA["O"], zA["B"])

CFG = dict(n_estimators=1000, num_leaves=63, learning_rate=0.03)
def mk(R, A):
    X = np.vstack([R, A]); y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    c = lgb.LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1, **CFG); c.fit(X, y); return c
def boot_auc(y, s, n=2000, seed=0):
    a = roc_auc_score(y, s); rng = np.random.default_rng(seed); m = len(y); arr = []
    for _ in range(n):
        idx = rng.integers(0, m, m); yi = y[idx]
        if yi.min() != yi.max(): arr.append(roc_auc_score(yi, s[idx]))
    lo, hi = np.percentile(arr, [2.5, 97.5]); return a, lo, hi

fl = list(FAMD); H = "Midjourney"
trRo = np.vstack([NAT[f][0] for f in fl if f != H]); trRb = np.vstack([NAT[f][1] for f in fl if f != H])
trAo = np.vstack([AI[f][0] for f in fl if f != H]);  trAb = np.vstack([AI[f][1] for f in fl if f != H])
opf = mk(trRo, trAo); bas = mk(trRb, trAb)
comb = mk(np.hstack([trRo, trRb]), np.hstack([trAo, trAb]))
y = np.r_[np.zeros(len(reO)), np.ones(len(aiO))]
res = {}
res["OPF-15"]  = boot_auc(y, opf.predict_proba(np.vstack([reO, aiO]))[:, 1])
res["Base-63"] = boot_auc(y, bas.predict_proba(np.vstack([reB, aiB]))[:, 1])
res["Fusion"]  = boot_auc(y, comb.predict_proba(np.vstack([np.hstack([reO, reB]), np.hstack([aiO, aiB])]))[:, 1])
out = [f"MODERN-GENERATOR ROW ({MODEL}, 512 PNG, prompts 'photo of {{class}}' 1:1 with MJ-fold reals;",
       f"detector: Midjourney LOGO fold, never saw these reals or this generator)  n={len(reO)}/{len(aiO)}"]
for k, (a, lo, hi) in res.items():
    out.append(f"  {k:8s} AUC {a:.3f} [{lo:.3f},{hi:.3f}]")
txt = "\n".join(out); print(txt, flush=True)
open(f"{AD}/{TAG}_row_results.txt", "w").write(txt + "\n")
print(f"WROTE {TAG}_row_results.txt", flush=True)
