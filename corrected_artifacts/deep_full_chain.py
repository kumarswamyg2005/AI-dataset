"""
MPS chain for the WIFS push (runs phases in order, each resumable):

PHASE 1  Materialize the normalized images once to data/norm256/:
         native 256 center crop + JPEG Q95 4:4:4 (identical to the feature pipeline),
         first 1500 valid images per family per class (same ordering as cache_more).
PHASE 2  Rescore NPR and FatFormer zero-shot on the FULL normalized sets
         (removes the 450-per-class subset caveat). Caches per-set .npy in
         deep_scores_full/; writes deep_full_results.txt.
PHASE 3  Fine-tune NPR on our fair protocol, leave-one-generator-out:
         init from released NPR.pth, train on the 5 other generators
         (random 224 crop + hflip from the normalized 256s), test on the held-out
         generator (center 224). Adam lr 1e-4, 4 epochs, batch 32, BCE.
         Per-fold checkpoint of results; writes npr_finetuned_results.txt.
"""
import os, sys, glob, time, random
import numpy as np, cv2, torch, argparse
from io import BytesIO
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE = "/Users/kumaraswamy/Desktop/jpg/research/full/baselines"
AD   = "/Users/kumaraswamy/Desktop/jpg/research/full/corrected_artifacts"
ROOT = "/Users/kumaraswamy/Desktop/jpg/research/data/tiny_genimage"
NORM = "/Users/kumaraswamy/Desktop/jpg/research/data/norm256"
sys.path.insert(0, BASE); sys.path.insert(0, f"{BASE}/fatformer")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
S, Q, SUB, NPC = 256, 95, 0, 1500
FAM = {"SD v1.5":"imagenet_ai_0424_sdv5","ADM":"imagenet_ai_0508_adm","GLIDE":"imagenet_glide",
       "Midjourney":"imagenet_midjourney","Wukong":"imagenet_ai_0424_wukong","VQDM":"imagenet_ai_0419_vqdm"}
torch.manual_seed(42); random.seed(42); np.random.seed(42)

# ---------------- PHASE 1: materialize ----------------
def norm_img(path):
    bgr = cv2.imread(str(path))
    if bgr is None: return None
    H, W = bgr.shape[:2]
    if H < S or W < S: return None
    t, l = (H-S)//2, (W-S)//2
    return Image.fromarray(cv2.cvtColor(bgr[t:t+S, l:l+S], cv2.COLOR_BGR2RGB))

def materialize(fam, cls):
    d = f"{NORM}/{FAM[fam]}/{cls}"; os.makedirs(d, exist_ok=True)
    done = f"{d}/.done"
    if os.path.exists(done): return sorted(glob.glob(f"{d}/*.jpg"))
    t0, n = time.time(), 0
    for p in sorted(glob.glob(f"{ROOT}/{FAM[fam]}/train/{cls}/*"))[:NPC+400]:
        if n >= NPC: break
        im = norm_img(p)
        if im is None: continue
        im.save(f"{d}/{os.path.splitext(os.path.basename(p))[0]}.jpg",
                format="JPEG", quality=Q, subsampling=SUB)
        n += 1
    open(done, "w").write("ok")
    print(f"[mat {fam} {cls}] {n} in {time.time()-t0:.0f}s", flush=True)
    return sorted(glob.glob(f"{d}/*.jpg"))

FILES = {(f, c): materialize(f, c) for f in FAM for c in ("nature", "ai")}
print("PHASE 1 done", flush=True)

# ---------------- shared: models & transforms ----------------
npr_te = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
npr_tr = transforms.Compose([transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ff_te = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711])])

def load_npr():
    from run_npr import NPRResNet
    m = NPRResNet(num_classes=1)
    raw = torch.load(f"{BASE}/checkpoints/NPR.pth", map_location="cpu")
    st = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    m.load_state_dict({k.replace("module.","",1): v for k, v in st.items()}, strict=True)
    return m

def load_ff():
    from models import build_model
    args = argparse.Namespace(backbone="CLIP:ViT-L/14", num_vit_adapter=3, num_context_embedding=8,
        init_context_embedding=None, num_classes=2)
    m = build_model(args)
    ck = torch.load(f"{BASE}/checkpoints/fatformer.pth", map_location="cpu")
    m.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
    return m

class ImgSet(Dataset):
    def __init__(self, items, tf):  # items: list of (path, label)
        self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), torch.tensor(float(y))

# ---------------- PHASE 2: full zero-shot rescoring ----------------
SD2 = f"{AD}/deep_scores_full"; os.makedirs(SD2, exist_ok=True)
def score_set(model, tf, kind, files, tag, bs=32):
    fp = f"{SD2}/{tag}.npy"
    if os.path.exists(fp): return np.load(fp)
    dl = DataLoader(ImgSet([(p, 0) for p in files], tf), batch_size=bs, num_workers=0)
    out = []; t0 = time.time()
    with torch.no_grad():
        for x, _ in dl:
            o = model(x.to(dev).float())
            s = torch.sigmoid(o).squeeze(1) if kind == "npr" else o.softmax(1)[:, 1]
            out.append(s.cpu().numpy())
    sc = np.concatenate(out); np.save(fp, sc)
    print(f"[score {tag}] n={len(sc)} in {time.time()-t0:.0f}s", flush=True)
    return sc

need2 = [(m, f, c) for m in ("npr", "fatformer") for f in FAM for c in ("nature", "ai")
         if not os.path.exists(f"{SD2}/{m}_{FAM[f]}_{c}.npy")]
if need2:
    for meth in ("npr", "fatformer"):
        todo = [x for x in need2 if x[0] == meth]
        if not todo: continue
        model = (load_npr() if meth == "npr" else load_ff()).to(dev).eval()
        for _, f, c in todo:
            score_set(model, npr_te if meth == "npr" else ff_te, meth,
                      FILES[(f, c)], f"{meth}_{FAM[f]}_{c}", bs=64 if meth == "npr" else 32)
        del model; torch.mps.empty_cache() if dev == "mps" else None
lines = ["DEEP ZERO-SHOT, FULL SETS (all normalized images, best-direction AUC)",
         f"{'family':12s} {'NPR':>7s} {'FatF':>7s}  n(re/ai)"]
nm, fm = [], []
for f in FAM:
    r = {m: np.load(f"{SD2}/{m}_{FAM[f]}_nature.npy") for m in ("npr", "fatformer")}
    a = {m: np.load(f"{SD2}/{m}_{FAM[f]}_ai.npy") for m in ("npr", "fatformer")}
    y = np.r_[np.zeros(len(r["npr"])), np.ones(len(a["npr"]))]
    aucs = {}
    for m in ("npr", "fatformer"):
        v = roc_auc_score(y, np.r_[r[m], a[m]]); aucs[m] = max(v, 1 - v)
    lines.append(f"{f:12s} {aucs['npr']:7.3f} {aucs['fatformer']:7.3f}  {len(r['npr'])}/{len(a['npr'])}")
    if f != "SD v1.5": nm.append(aucs["npr"]); fm.append(aucs["fatformer"])
lines.append(f"{'mean 5 uns.':12s} {np.mean(nm):7.3f} {np.mean(fm):7.3f}")
open(f"{AD}/deep_full_results.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True); print("PHASE 2 done", flush=True)

# ---------------- PHASE 3: NPR fine-tune, leave-one-generator-out ----------------
RES3 = f"{AD}/npr_ft_folds"; os.makedirs(RES3, exist_ok=True)
def finetune_fold(H):
    fp = f"{RES3}/{FAM[H]}.txt"
    if os.path.exists(fp): return float(open(fp).read().strip())
    tr = [(p, 0) for f in FAM if f != H for p in FILES[(f, "nature")]] + \
         [(p, 1) for f in FAM if f != H for p in FILES[(f, "ai")]]
    random.Random(42).shuffle(tr)
    dl = DataLoader(ImgSet(tr, npr_tr), batch_size=32, shuffle=True, num_workers=0)
    m = load_npr().to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(4):
        t0, tot = time.time(), 0.0
        for x, y in dl:
            opt.zero_grad()
            out = m(x.to(dev).float()).squeeze(1)
            loss = lossf(out, y.to(dev)); loss.backward(); opt.step()
            tot += loss.item() * len(y)
        print(f"[npr-ft {H}] epoch {ep+1}/4 loss {tot/len(tr):.4f} in {time.time()-t0:.0f}s", flush=True)
    m.eval(); scores, ys = [], []
    te = [(p, 0) for p in FILES[(H, "nature")]] + [(p, 1) for p in FILES[(H, "ai")]]
    with torch.no_grad():
        for x, y in DataLoader(ImgSet(te, npr_te), batch_size=64, num_workers=0):
            scores.append(torch.sigmoid(m(x.to(dev).float()).squeeze(1)).cpu().numpy())
            ys.append(y.numpy())
    auc = roc_auc_score(np.concatenate(ys), np.concatenate(scores))
    open(fp, "w").write(f"{auc:.4f}\n")
    del m; torch.mps.empty_cache() if dev == "mps" else None
    return auc

out3 = ["NPR FINE-TUNED on our fair protocol (LOGO: init NPR.pth, 4 ep, lr 1e-4, b32,",
        "rand224+flip from normalized 256s; test = held-out generator, center 224)",
        f"{'held-out':12s} {'NPR-ft AUC':>10s}"]
vals = []
for H in FAM:
    a = finetune_fold(H)
    out3.append(f"{H:12s} {a:10.3f}"); print(out3[-1], flush=True)
    if H != "SD v1.5": vals.append(a)
out3.append(f"{'mean 5 uns.':12s} {np.mean(vals):10.3f}")
open(f"{AD}/npr_finetuned_results.txt", "w").write("\n".join(out3) + "\n")
print("\n".join(out3), flush=True); print("PHASE 3 done - CHAIN COMPLETE", flush=True)
