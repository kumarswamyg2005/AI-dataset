"""
MODERN-GENERATOR ROW: SDXL-Turbo (2024) under the fair protocol.

Design (no leakage, content-matched):
  - Fakes: SDXL-Turbo 512x512, one image per prompt "photo of {class}", where the classes are
    taken 1:1 from the first N materialized Midjourney-fold real images (same WNID histogram).
    Saved as PNG (born digital, like GenImage fakes).
  - Reals: those same N Midjourney-family nature images.
  - Detector: the Midjourney LOGO fold (trained on SD v1.5, ADM, GLIDE, Wukong, VQDM) -- this
    model has never seen Midjourney reals nor, of course, SDXL-Turbo fakes.
  - Same normalization (native 256 crop, JPEG Q95 4:4:4), same tuned LightGBM config.
Writes sdxl_row_results.txt; PNGs in data/sdxl_turbo/ai; features cached in cache_sdxl/.
"""
import os, sys, glob, json, time
import numpy as np, cv2, torch
from io import BytesIO
from PIL import Image
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, "/Users/kumaraswamy/Desktop/jpg/research/full")
import opf_local_train as L

AD   = "/Users/kumaraswamy/Desktop/jpg/research/full/corrected_artifacts"
NORM = "/Users/kumaraswamy/Desktop/jpg/research/data/norm256"
# model is switchable: SDXL-Turbo needs a >=16GB GPU (Kaggle); SD-Turbo fits the local 8GB Mac
MODEL = os.environ.get("GEN_MODEL", "stabilityai/sdxl-turbo")
TAG   = os.environ.get("GEN_TAG", "sdxl_turbo")
GEN  = f"/Users/kumaraswamy/Desktop/jpg/research/data/{TAG}/ai"
CIX  = "/Users/kumaraswamy/Desktop/jpg/research/data/imagenet_class_index.json"
os.makedirs(GEN, exist_ok=True)
N = 1000
S, Q, SUB = 256, 95, 0
FAMD = {"SD v1.5":"imagenet_ai_0424_sdv5","ADM":"imagenet_ai_0508_adm","GLIDE":"imagenet_glide",
        "Midjourney":"imagenet_midjourney","Wukong":"imagenet_ai_0424_wukong","VQDM":"imagenet_ai_0419_vqdm"}

w2n = {v[0]: v[1].replace("_", " ") for v in json.load(open(CIX)).values()}
mj_reals = sorted(glob.glob(f"{NORM}/imagenet_midjourney/nature/*.jpg"))[:N]
assert len(mj_reals) == N, f"need {N} materialized MJ reals, found {len(mj_reals)}"
classes = [w2n[os.path.basename(p).split("_")[0]] for p in mj_reals]

# ---------- generation (resumable per image) ----------
todo = [(i, c) for i, c in enumerate(classes) if not os.path.exists(f"{GEN}/{i:04d}.png")]
if todo:
    from diffusers import AutoPipelineForText2Image
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16").to(dev)
    pipe.set_progress_bar_config(disable=True)
    t0 = time.time()
    for k, (i, c) in enumerate(todo):
        g = torch.Generator("cpu").manual_seed(i)
        img = pipe(prompt=f"photo of {c}", num_inference_steps=2, guidance_scale=0.0,
                   height=512, width=512, generator=g).images[0]
        img.save(f"{GEN}/{i:04d}.png")
        if (k + 1) % 50 == 0:
            r = (time.time() - t0) / (k + 1)
            print(f"[gen] {k+1}/{len(todo)} ({r:.1f}s/img, ~{r*(len(todo)-k-1)/60:.0f} min left)", flush=True)
    del pipe
    if dev == "mps": torch.mps.empty_cache()
print(f"generation complete: {len(glob.glob(f'{GEN}/*.png'))} images", flush=True)

# ---------- features under the identical pipeline ----------
def norm_bgr(path):
    bgr = cv2.imread(str(path))
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

CD = f"{AD}/cache_{TAG}"; os.makedirs(CD, exist_ok=True)
def feats(paths, key, renorm):
    fp = f"{CD}/{key}.npz"
    if os.path.exists(fp):
        z = np.load(fp); return z["O"], z["B"]
    O, B = [], []
    for p in paths:
        # reals in norm256/ are already normalized; fakes need the full pipeline
        bgr = norm_bgr(p) if renorm else cv2.imread(str(p))
        if bgr is None: continue
        of = L._extract_from_bgr(bgr)
        bf = spec63(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32))
        if of is not None and np.all(np.isfinite(of)) and np.all(np.isfinite(bf)):
            O.append(of); B.append(bf)
    O, B = np.array(O, np.float32), np.array(B, np.float32)
    np.savez(fp, O=O, B=B); print(f"[feat {key}] {len(O)}", flush=True)
    return O, B

reO, reB = feats(mj_reals, "mj_reals", renorm=False)
aiO, aiB = feats(sorted(glob.glob(f"{GEN}/*.png")), f"{TAG}_ai", renorm=True)

# ---------- evaluate with the Midjourney LOGO fold ----------
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
