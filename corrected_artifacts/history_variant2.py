"""
Second compression-history variant (robustness of paper finding 4).
Variant 2: prior JPEG at LOWER quality range Q in [70,90], encoded with cv2/libjpeg
(different encoder + different quantization tables than variant 1's PIL Q85-98).
Probe only: models trained on STANDARD features, tested vs variant-2 history fakes.
Writes history_variant2_results.txt; caches fakes to cache_jctl2/.
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
CD = f"{AD}/cache_jctl2"; os.makedirs(CD, exist_ok=True)

def prior_q(path):  # deterministic Q in [70,90]
    h = int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16)
    return 70 + (h % 21)

def jpeg_cycle_cv2(bgr, q):  # libjpeg encoder via OpenCV
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else None

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

def extract(fam):
    fp = f"{CD}/{FAM[fam]}_ai.npz"
    if os.path.exists(fp):
        z = np.load(fp); return z["O"], z["B"]
    O, B = [], []; t0 = time.time()
    for p in gf(fam, "ai")[:NPC+400]:
        if len(O) >= NPC: break
        bgr = cv2.imread(str(p))
        if bgr is None: continue
        bgr = jpeg_cycle_cv2(bgr, prior_q(p))
        n = norm(bgr)
        if n is None: continue
        of = L._extract_from_bgr(n)
        bf = spec63(cv2.cvtColor(n, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32))
        if of is not None and np.all(np.isfinite(of)) and np.all(np.isfinite(bf)):
            O.append(of); B.append(bf)
    O, B = np.array(O, np.float32), np.array(B, np.float32)
    np.savez(fp, O=O, B=B); print(f"[jctl2 {fam}] {len(O)} in {time.time()-t0:.0f}s", flush=True)
    return O, B

AIJ2 = {f: extract(f) for f in FAM}

NAT, AI = {}, {}
for f, d in FAM.items():
    zN = np.load(f"{AD}/cache_more/{d}_nature.npz"); zA = np.load(f"{AD}/cache_more/{d}_ai.npz")
    NAT[f] = (zN["O"], zN["B"]); AI[f] = (zA["O"], zA["B"])
CFG = dict(n_estimators=1000, num_leaves=63, learning_rate=0.03)
def mk(R, A):
    X = np.vstack([R, A]); y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    c = lgb.LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1, **CFG); c.fit(X, y); return c
def auc_of(clf, R, A):
    X = np.vstack([R, A]); y = np.r_[np.zeros(len(R)), np.ones(len(A))]
    return roc_auc_score(y, clf.predict_proba(X)[:, 1])

fl = list(FAM)
out = ["HISTORY VARIANT 2 (cv2/libjpeg encoder, prior Q in [70,90]) - probe with std-trained models",
       f"{'held-out':12s} {'B63 std':>8s} {'B63 v2':>7s} {'drop':>7s} | {'OPF std':>8s} {'OPF v2':>7s} {'drop':>7s}"]
db, do = [], []
for H in fl:
    trRb = np.vstack([NAT[f][1] for f in fl if f != H]); trAb = np.vstack([AI[f][1] for f in fl if f != H])
    trRo = np.vstack([NAT[f][0] for f in fl if f != H]); trAo = np.vstack([AI[f][0] for f in fl if f != H])
    bas = mk(trRb, trAb); opf = mk(trRo, trAo)
    b_s = auc_of(bas, NAT[H][1], AI[H][1]);  b_2 = auc_of(bas, NAT[H][1], AIJ2[H][1])
    o_s = auc_of(opf, NAT[H][0], AI[H][0]);  o_2 = auc_of(opf, NAT[H][0], AIJ2[H][0])
    out.append(f"{H:12s} {b_s:8.3f} {b_2:7.3f} {b_2-b_s:+7.3f} | {o_s:8.3f} {o_2:7.3f} {o_2-o_s:+7.3f}")
    db.append(b_2 - b_s); do.append(o_2 - o_s)
out.append(f"{'mean drop':12s} {'':8s} {'':7s} {np.mean(db):+7.3f} | {'':8s} {'':7s} {np.mean(do):+7.3f}")
txt = "\n".join(out); print("\n" + txt, flush=True)
open(f"{AD}/history_variant2_results.txt", "w").write(txt + "\n")
print("WROTE history_variant2_results.txt", flush=True)
