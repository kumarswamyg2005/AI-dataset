"""Verification of main_corrected.tex numbers against corrected_artifacts."""
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

AD = "/Users/kumaraswamy/Desktop/jpg/research/full/corrected_artifacts"
FAM = {"SD v1.5":"imagenet_ai_0424_sdv5","ADM":"imagenet_ai_0508_adm","GLIDE":"imagenet_glide",
       "Midjourney":"imagenet_midjourney","Wukong":"imagenet_ai_0424_wukong","VQDM":"imagenet_ai_0419_vqdm"}
NAT, AI = {}, {}
print("== cache_more counts (nature / ai) ==")
for f, d in FAM.items():
    zN = np.load(f"{AD}/cache_more/{d}_nature.npz"); zA = np.load(f"{AD}/cache_more/{d}_ai.npz")
    NAT[f] = (zN["O"], zN["B"]); AI[f] = (zA["O"], zA["B"])
    print(f"{f:12s} {len(zN['O']):5d} / {len(zA['O']):5d}")

# ---- Cohen's d on pooled real vs AI (paper: Cr 0.15, Cb 0.13, Y 0.10, '9000 images each') ----
natO = np.vstack([NAT[f][0] for f in FAM]); aiO = np.vstack([AI[f][0] for f in FAM])
print(f"\n== Cohen's d, pooled n_real={len(natO)} n_ai={len(aiO)} (features 0,1,2 = rho(eL,eH) Y/Cb/Cr) ==")
def cohend(a, b):
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / sp
for i, nm in [(0, "Y"), (1, "Cb"), (2, "Cr")]:
    d = cohend(natO[:, i], aiO[:, i])
    print(f"  {nm:3s} d = {d:+.4f}  (|d| = {abs(d):.3f})")

# ---- Deep detector AUCs from saved scores (paper Table I NPR / FatFormer cols) ----
print("\n== Deep zero-shot AUCs (best direction) from deep_scores/*.npy ==")
GENS = ["SDv1.5", "ADM", "GLIDE", "Midjourney", "Wukong", "VQDM"]
for det in ["npr", "fatformer"]:
    m5 = []
    for g in GENS:
        r = np.load(f"{AD}/deep_scores/{det}_{g}_nature.npy")
        a = np.load(f"{AD}/deep_scores/{det}_{g}_ai.npy")
        y = np.r_[np.zeros(len(r)), np.ones(len(a))]; s = np.r_[r, a]
        auc = roc_auc_score(y, s); auc = max(auc, 1 - auc)
        print(f"  {det:9s} {g:11s} {auc:.3f}   n={len(r)}/{len(a)}")
        if g != "SDv1.5": m5.append(auc)
    print(f"  {det:9s} mean(5 unseen) {np.mean(m5):.4f}")

# ---- Reproduce Table I OPF/Trivial/Fusion: LOGO with paper hyperparams (1000 trees, 63 leaves, lr .03) ----
def mk(R, A, **kw):
    X = np.vstack([R, A]); y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    c = lgb.LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1, **kw)
    c.fit(X, y); return c
def get_auc(clf, R, A):
    X = np.vstack([R, A]); y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    return roc_auc_score(y, clf.predict_proba(X)[:, 1])

PAPER = {"SD v1.5": (0.838, 0.924, 0.963), "ADM": (0.723, 0.883, 0.883),
         "GLIDE": (0.711, 0.941, 0.932), "Midjourney": (0.820, 0.869, 0.861),
         "Wukong": (0.762, 0.894, 0.938), "VQDM": (0.682, 0.915, 0.917)}
fams = list(FAM)
CFG = dict(n_estimators=1000, num_leaves=63, learning_rate=0.03)
print(f"\n== LOGO reproduction with {CFG} (paper values in parens) ==")
mo, mb, mc = [], [], []
for H in fams:
    trRo = np.vstack([NAT[f][0] for f in fams if f != H]); trRb = np.vstack([NAT[f][1] for f in fams if f != H])
    trAo = np.vstack([AI[f][0] for f in fams if f != H]);  trAb = np.vstack([AI[f][1] for f in fams if f != H])
    opf = mk(trRo, trAo, **CFG); bas = mk(trRb, trAb, **CFG)
    comb = mk(np.hstack([trRo, trRb]), np.hstack([trAo, trAb]), **CFG)
    reO, reB = NAT[H]; aO, aB = AI[H]
    ao = get_auc(opf, reO, aO); ab = get_auc(bas, reB, aB)
    ac = get_auc(comb, np.hstack([reO, reB]), np.hstack([aO, aB]))
    p = PAPER[H]
    print(f"  {H:12s} OPF {ao:.3f} ({p[0]:.3f})   Base {ab:.3f} ({p[1]:.3f})   Comb {ac:.3f} ({p[2]:.3f})")
    if H != "SD v1.5": mo.append(ao); mb.append(ab); mc.append(ac)
print(f"  {'mean 5 unseen':12s} OPF {np.mean(mo):.3f} (0.740)   Base {np.mean(mb):.3f} (0.900)   Comb {np.mean(mc):.3f} (0.906)")

# ---- Table arithmetic: do the printed averages match the printed rows? ----
print("\n== Table I column means recomputed from the paper's own printed row values ==")
for j, nm, tgt in [(0, "OPF", 0.740), (1, "Trivial", 0.900), (2, "Fusion", 0.906)]:
    vals = [PAPER[g][j] for g in fams if g != "SD v1.5"]
    print(f"  {nm:8s} mean = {np.mean(vals):.4f}  (paper prints {tgt})")
for det, tgt, vals in [("NPR", 0.630, [0.708, 0.652, 0.668, 0.576, 0.546]),
                       ("FatF", 0.844, [0.941, 0.857, 0.613, 0.867, 0.941])]:
    print(f"  {det:8s} mean = {np.mean(vals):.4f}  (paper prints {tgt})")
