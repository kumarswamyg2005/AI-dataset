"""
Two follow-up experiments on the corrected, bias-controlled protocol (native-256 crop, Q95, 4:4:4):

1) JPEG-HISTORY CONTROL. In tiny_genimage every real is a prior JPEG (ImageNet .JPEG) and every
   fake is PNG, so after the shared Q95 re-encode the classes still differ in compression history
   (double vs single). Control: give each fake a synthetic prior JPEG life (random Q in [85,98],
   deterministic per path) BEFORE the standard pipeline, so both classes carry a JPEG history.
   (a) probe: models trained on standard features, tested with history-matched fakes;
   (b) full control: leave-one-generator-out retrained with history-matched fakes in train+test.

2) BASE-189. The trivial baseline extended to all three channels (63 log-var spectrum numbers for
   each of Y, Cr, Cb), same classifier, leave-one-generator-out. Accuracy-improvement candidate.

Classifier everywhere: LightGBM(n_estimators=1000, num_leaves=63, learning_rate=0.03,
class_weight='balanced', random_state=42) — the paper's tuned configuration.
Caches: cache_jpegctl/ (per family, fakes only), cache_base189/ (per family & class).
"""
import os, sys, glob, time, hashlib
import numpy as np, cv2
from io import BytesIO
from PIL import Image
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, "/Users/kumaraswamy/Desktop/jpg/research/full")
import opf_local_train as L

AD   = "/Users/kumaraswamy/Desktop/jpg/research/full/corrected_artifacts"
ROOT = "/Users/kumaraswamy/Desktop/jpg/research/data/tiny_genimage"
S, Q, SUB, NPC = 256, 95, 0, 1500
FAM = {"SD v1.5":"imagenet_ai_0424_sdv5","ADM":"imagenet_ai_0508_adm","GLIDE":"imagenet_glide",
       "Midjourney":"imagenet_midjourney","Wukong":"imagenet_ai_0424_wukong","VQDM":"imagenet_ai_0419_vqdm"}
C1 = f"{AD}/cache_jpegctl";  os.makedirs(C1, exist_ok=True)
C2 = f"{AD}/cache_base189"; os.makedirs(C2, exist_ok=True)

def prior_q(path):  # deterministic Q in [85,98] per file
    h = int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16)
    return 85 + (h % 14)

def jpeg_cycle(bgr, q):
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    buf = BytesIO(); im.save(buf, format="JPEG", quality=q); buf.seek(0)
    return cv2.cvtColor(np.array(Image.open(buf).convert("RGB")), cv2.COLOR_RGB2BGR)

def norm(bgr):
    if bgr is None: return None
    H, W = bgr.shape[:2]
    if H < S or W < S: return None
    t, l = (H-S)//2, (W-S)//2
    im = Image.fromarray(cv2.cvtColor(bgr[t:t+S, l:l+S], cv2.COLOR_BGR2RGB))
    buf = BytesIO(); im.save(buf, format="JPEG", quality=Q, subsampling=SUB); buf.seek(0)
    return cv2.cvtColor(np.array(Image.open(buf).convert("RGB")), cv2.COLOR_RGB2BGR)

def spec63(y):
    H, W = y.shape; nh, nw = H//8, W//8
    acc = np.empty((nh*nw, 64), np.float32); b = 0
    for r in range(nh):
        for c in range(nw): acc[b] = cv2.dct(y[r*8:r*8+8, c*8:c*8+8]).flatten(); b += 1
    return np.log1p(acc.var(axis=0)[1:]).astype(np.float32)

def gf(fam, cls): return sorted(glob.glob(f"{ROOT}/{FAM[fam]}/train/{cls}/*"))

# ---------- extraction phase ----------
def extract_jpegctl(fam):
    fp = f"{C1}/{FAM[fam]}_ai.npz"
    if os.path.exists(fp):
        z = np.load(fp); return z["O"], z["B"]
    O, B = [], []; t0 = time.time()
    for p in gf(fam, "ai")[:NPC+400]:
        if len(O) >= NPC: break
        bgr = cv2.imread(str(p))
        if bgr is None: continue
        bgr = jpeg_cycle(bgr, prior_q(p))
        n = norm(bgr)
        if n is None: continue
        of = L._extract_from_bgr(n)
        bf = spec63(cv2.cvtColor(n, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32))
        if of is not None and np.all(np.isfinite(of)) and np.all(np.isfinite(bf)):
            O.append(of); B.append(bf)
    O, B = np.array(O, np.float32), np.array(B, np.float32)
    np.savez(fp, O=O, B=B)
    print(f"[jpegctl {fam}] {len(O)} in {time.time()-t0:.0f}s", flush=True)
    return O, B

def extract_b189(fam, cls, jctl=False):
    fp = f"{C2}/{FAM[fam]}_{cls}{'_jctl' if jctl else ''}.npz"
    if os.path.exists(fp):
        return np.load(fp)["F"]
    F = []; t0 = time.time()
    for p in gf(fam, cls)[:NPC+400]:
        if len(F) >= NPC: break
        bgr = cv2.imread(str(p))
        if bgr is None: continue
        if jctl: bgr = jpeg_cycle(bgr, prior_q(p))
        n = norm(bgr)
        if n is None: continue
        ycc = cv2.cvtColor(n, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        f = np.hstack([spec63(ycc[:, :, k]) for k in range(3)])
        if np.all(np.isfinite(f)): F.append(f)
    F = np.array(F, np.float32); np.savez(fp, F=F)
    print(f"[b189 {fam} {cls}{' jctl' if jctl else ''}] {len(F)} in {time.time()-t0:.0f}s", flush=True)
    return F

AI_J = {f: extract_jpegctl(f) for f in FAM}                      # fakes, history-matched, OPF15+63
B189 = {f: {"nature": extract_b189(f, "nature"), "ai": extract_b189(f, "ai")} for f in FAM}
print("extraction done", flush=True)

# ---------- standard cached features ----------
NAT, AI = {}, {}
for f, d in FAM.items():
    zN = np.load(f"{AD}/cache_more/{d}_nature.npz"); zA = np.load(f"{AD}/cache_more/{d}_ai.npz")
    NAT[f] = (zN["O"], zN["B"]); AI[f] = (zA["O"], zA["B"])

CFG = dict(n_estimators=1000, num_leaves=63, learning_rate=0.03)
def mk(R, A):
    X = np.vstack([R, A]); y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    c = lgb.LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1, **CFG)
    c.fit(X, y); return c
def auc_of(clf, R, A):
    X = np.vstack([R, A]); y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    return roc_auc_score(y, clf.predict_proba(X)[:, 1])

fl = list(FAM)
out = ["JPEG-HISTORY CONTROL + BASE-189 (tuned cfg 1000/63/0.03, LOGO, native256 Q95 4:4:4)", ""]

# ---- 1a) probe: standard-trained models, history-matched test fakes ----
out.append("1a) PROBE - trained on standard fakes, tested vs history-matched fakes (Base-63):")
out.append(f"{'held-out':12s} {'std AUC':>8s} {'jctl AUC':>9s} {'drop':>7s}")
for H in fl:
    trR = np.vstack([NAT[f][1] for f in fl if f != H]); trA = np.vstack([AI[f][1] for f in fl if f != H])
    bas = mk(trR, trA)
    a_std = auc_of(bas, NAT[H][1], AI[H][1])
    a_j   = auc_of(bas, NAT[H][1], AI_J[H][1])
    out.append(f"{H:12s} {a_std:8.3f} {a_j:9.3f} {a_j-a_std:+7.3f}")
out.append("")

# ---- 1b) full control: history-matched fakes in train AND test ----
out.append("1b) FULL CONTROL - history-matched fakes in train and test:")
out.append(f"{'held-out':12s} {'OPF-15':>7s} {'Base-63':>8s} {'Fusion':>7s}")
mo, mb, mc = [], [], []
for H in fl:
    trRo = np.vstack([NAT[f][0] for f in fl if f != H]); trRb = np.vstack([NAT[f][1] for f in fl if f != H])
    trAo = np.vstack([AI_J[f][0] for f in fl if f != H]); trAb = np.vstack([AI_J[f][1] for f in fl if f != H])
    opf = mk(trRo, trAo); bas = mk(trRb, trAb)
    comb = mk(np.hstack([trRo, trRb]), np.hstack([trAo, trAb]))
    ao = auc_of(opf, NAT[H][0], AI_J[H][0]); ab = auc_of(bas, NAT[H][1], AI_J[H][1])
    ac = auc_of(comb, np.hstack(NAT[H]), np.hstack(AI_J[H]))
    out.append(f"{H:12s} {ao:7.3f} {ab:8.3f} {ac:7.3f}")
    if H != "SD v1.5": mo.append(ao); mb.append(ab); mc.append(ac)
out.append(f"{'mean 5 uns.':12s} {np.mean(mo):7.3f} {np.mean(mb):8.3f} {np.mean(mc):7.3f}")
out.append("")

# ---- 2) Base-189 (3-channel spectrum), standard protocol ----
out.append("2) BASE-189 (Y+Cr+Cb spectrum) vs Base-63, standard protocol:")
out.append(f"{'held-out':12s} {'Base-63':>8s} {'Base-189':>9s} {'Fus15+189':>10s}")
m63, m189, mf = [], [], []
for H in fl:
    trR63 = np.vstack([NAT[f][1] for f in fl if f != H]); trA63 = np.vstack([AI[f][1] for f in fl if f != H])
    trR189 = np.vstack([B189[f]["nature"] for f in fl if f != H]); trA189 = np.vstack([B189[f]["ai"] for f in fl if f != H])
    # align counts for fusion (independent extraction passes can differ by a few kept images)
    trRo = np.vstack([NAT[f][0] for f in fl if f != H]); trAo = np.vstack([AI[f][0] for f in fl if f != H])
    b63 = mk(trR63, trA63); b189 = mk(trR189, trA189)
    a63 = auc_of(b63, NAT[H][1], AI[H][1]); a189 = auc_of(b189, B189[H]["nature"], B189[H]["ai"])
    nR = min(len(trRo), len(trR189)); nA = min(len(trAo), len(trA189))
    nRt = min(len(NAT[H][0]), len(B189[H]["nature"])); nAt = min(len(AI[H][0]), len(B189[H]["ai"]))
    fus = mk(np.hstack([trRo[:nR], trR189[:nR]]), np.hstack([trAo[:nA], trA189[:nA]]))
    af = auc_of(fus, np.hstack([NAT[H][0][:nRt], B189[H]["nature"][:nRt]]),
                np.hstack([AI[H][0][:nAt], B189[H]["ai"][:nAt]]))
    out.append(f"{H:12s} {a63:8.3f} {a189:9.3f} {af:10.3f}")
    if H != "SD v1.5": m63.append(a63); m189.append(a189); mf.append(af)
out.append(f"{'mean 5 uns.':12s} {np.mean(m63):8.3f} {np.mean(m189):9.3f} {np.mean(mf):10.3f}")

txt = "\n".join(out)
print("\n" + txt, flush=True)
open(f"{AD}/jpeg_control_and_base189_results.txt", "w").write(txt + "\n")
print("\nWROTE jpeg_control_and_base189_results.txt", flush=True)
