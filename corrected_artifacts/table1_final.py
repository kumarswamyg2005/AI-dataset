"""
FINAL Table I artifact for main_corrected.tex (regenerated with the paper's tuned configuration).

Protocol (identical to the paper):
  - genuine content-matched real(ImageNet nature)-vs-AI(GenImage), native-256 center crop,
    re-encoded JPEG Q95 4:4:4, both classes identically; features from cache_more (1500/class/gen).
  - UNSEEN rows: leave-one-generator-out (held-out generator never in training).
  - IN-DOMAIN row: train on the five other generators (full pools) + the SD v1.5 TRAIN split
    (cache_v2/tr_*, paths[:1500]); test on the disjoint SD v1.5 TEST split (cache_v2/te_*,
    paths[1500:2000]). SD v1.5 is therefore genuinely SEEN in training for this row only.
  - classifier: LightGBM(1000 trees, 63 leaves, lr 0.03, balanced, seed 42).
  - AUC (AI positive) with 95% bootstrap CI (2000 resamples).
Also reports a score-average fusion variant (mean of the two models' scores) as a diagnostic.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

AD = "/Users/kumaraswamy/Desktop/jpg/research/full/corrected_artifacts"
FAM = {"SD v1.5":"imagenet_ai_0424_sdv5","ADM":"imagenet_ai_0508_adm","GLIDE":"imagenet_glide",
       "Midjourney":"imagenet_midjourney","Wukong":"imagenet_ai_0424_wukong","VQDM":"imagenet_ai_0419_vqdm"}
NAT, AI = {}, {}
for f, d in FAM.items():
    zN = np.load(f"{AD}/cache_more/{d}_nature.npz"); zA = np.load(f"{AD}/cache_more/{d}_ai.npz")
    NAT[f] = (zN["O"], zN["B"]); AI[f] = (zA["O"], zA["B"])
def cv2f(name):
    z = np.load(f"{AD}/cache_v2/{name}.npz"); return z["O"], z["B"]

CFG = dict(n_estimators=1000, num_leaves=63, learning_rate=0.03)
def mk(R, A):
    X = np.vstack([R, A]); y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    c = lgb.LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1, **CFG)
    c.fit(X, y); return c
def boot(y, s, n=2000, seed=0):
    a = roc_auc_score(y, s); rng = np.random.default_rng(seed); m = len(y); arr = []
    for _ in range(n):
        idx = rng.integers(0, m, m); yi = y[idx]
        if yi.min() != yi.max(): arr.append(roc_auc_score(yi, s[idx]))
    lo, hi = np.percentile(arr, [2.5, 97.5]); return a, lo, hi
def scores(clf, R, A):
    y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    return y, clf.predict_proba(np.vstack([R, A]))[:, 1]

fl = list(FAM)
out = ["TABLE 1 FINAL (tuned cfg: 1000 trees, 63 leaves, lr 0.03 | LOGO | boot 2000)",
       f"{'test gen':12s} {'OPF-15 [95% CI]':>22s} {'Base-63 [95% CI]':>22s} {'Fusion-78 [95% CI]':>22s} {'ScoreAvg':>9s}  n(re/ai)"]
mo, mb, mc, ms, hw = [], [], [], [], []

# ---- in-domain row: SD v1.5 genuinely seen in training ----
trR = [NAT[f] for f in fl if f != "SD v1.5"] + [cv2f("tr_nature")]
trA = [AI[f] for f in fl if f != "SD v1.5"] + [cv2f("tr_ai")]
teR = cv2f("te_nature_indom"); teA = cv2f("te_ai_indom")
opf = mk(np.vstack([t[0] for t in trR]), np.vstack([t[0] for t in trA]))
bas = mk(np.vstack([t[1] for t in trR]), np.vstack([t[1] for t in trA]))
comb = mk(np.vstack([np.hstack(t) for t in trR]), np.vstack([np.hstack(t) for t in trA]))
yo, so = scores(opf, teR[0], teA[0]); yb, sb = scores(bas, teR[1], teA[1])
yc, sc = scores(comb, np.hstack(teR), np.hstack(teA))
ro, rb, rc = boot(yo, so), boot(yb, sb), boot(yc, sc)
sa = boot(yo, (so + sb) / 2)
hw += [(r[2]-r[1])/2 for r in (ro, rb, rc)]
out.append(f"{'SDv1.5 SEEN':12s} {ro[0]:.3f}[{ro[1]:.3f},{ro[2]:.3f}] {rb[0]:.3f}[{rb[1]:.3f},{rb[2]:.3f}] "
           f"{rc[0]:.3f}[{rc[1]:.3f},{rc[2]:.3f}] {sa[0]:9.3f}  {len(teR[0])}/{len(teA[0])}")

# ---- unseen rows: leave-one-generator-out ----
for H in fl:
    trRo = np.vstack([NAT[f][0] for f in fl if f != H]); trRb = np.vstack([NAT[f][1] for f in fl if f != H])
    trAo = np.vstack([AI[f][0] for f in fl if f != H]);  trAb = np.vstack([AI[f][1] for f in fl if f != H])
    opf = mk(trRo, trAo); bas = mk(trRb, trAb)
    comb = mk(np.hstack([trRo, trRb]), np.hstack([trAo, trAb]))
    yo, so = scores(opf, NAT[H][0], AI[H][0]); yb, sb = scores(bas, NAT[H][1], AI[H][1])
    yc, sc = scores(comb, np.hstack(NAT[H]), np.hstack(AI[H]))
    ro, rb, rc = boot(yo, so), boot(yb, sb), boot(yc, sc)
    sa = boot(yo, (so + sb) / 2)
    hw += [(r[2]-r[1])/2 for r in (ro, rb, rc)]
    out.append(f"{H+' unseen':12s} {ro[0]:.3f}[{ro[1]:.3f},{ro[2]:.3f}] {rb[0]:.3f}[{rb[1]:.3f},{rb[2]:.3f}] "
               f"{rc[0]:.3f}[{rc[1]:.3f},{rc[2]:.3f}] {sa[0]:9.3f}  {len(NAT[H][0])}/{len(AI[H][0])}")
    if H != "SD v1.5": mo.append(ro[0]); mb.append(rb[0]); mc.append(rc[0]); ms.append(sa[0])

out.append(f"{'mean 5 uns.':12s} {np.mean(mo):.3f}{'':17s} {np.mean(mb):.3f}{'':17s} "
           f"{np.mean(mc):.3f}{'':17s} {np.mean(ms):9.3f}")
out.append(f"max CI half-width, our methods: {max(hw):.4f}")
txt = "\n".join(out)
print(txt)
open(f"{AD}/table1_final_results.txt", "w").write(txt + "\n")
print("\nWROTE table1_final_results.txt")
