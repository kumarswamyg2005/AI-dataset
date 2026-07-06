"""
Deep baselines (NPR, FatFormer) zero-shot on the SAME corrected, content-matched, bias-controlled
data as OPF/Base: real=nature vs ai per family, native-256 crop + Q95 4:4:4 (identical inputs).
Best-direction AUC (score inversion if <0.5), matching the paper's convention.
Per-(method,family,class) score caching -> resumable across foreground calls. Device: MPS.
"""
import sys, os, glob, time, numpy as np, torch, argparse
from io import BytesIO
from PIL import Image
import cv2
from torchvision import transforms
from sklearn.metrics import roc_auc_score
BASE = "/Users/kumaraswamy/Desktop/jpg/research/full/baselines"
ROOT = "/Users/kumaraswamy/Desktop/jpg/research/data/tiny_genimage"
SC   = "/private/tmp/claude-501/-Users-kumaraswamy-Desktop-jpg-research/eb427cd5-6ebc-464d-a807-d7deeec6f45e/scratchpad"
CDIR = f"{SC}/deep_scores"; os.makedirs(CDIR, exist_ok=True)
sys.path.insert(0, BASE); sys.path.insert(0, f"{BASE}/fatformer")
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
S, Q, SUB, CAP = 256, 95, 0, 450
FAM = {"SD v1.5":"imagenet_ai_0424_sdv5","ADM":"imagenet_ai_0508_adm","GLIDE":"imagenet_glide",
       "Midjourney":"imagenet_midjourney","Wukong":"imagenet_ai_0424_wukong","VQDM":"imagenet_ai_0419_vqdm"}

def norm_pil(path):
    bgr = cv2.imread(str(path))
    if bgr is None: return None
    H, W = bgr.shape[:2]
    if H < S or W < S: return None
    t, l = (H-S)//2, (W-S)//2
    im = Image.fromarray(cv2.cvtColor(bgr[t:t+S, l:l+S], cv2.COLOR_BGR2RGB))
    buf = BytesIO(); im.save(buf, format="JPEG", quality=Q, subsampling=SUB); buf.seek(0)
    return Image.open(buf).convert("RGB")

def gf(fam, cls): return sorted(glob.glob(f"{ROOT}/{FAM[fam]}/train/{cls}/*"))

npr_tf = transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ff_tf = transforms.Compose([transforms.Resize((224,224)), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711])])

def load_npr():
    from run_npr import NPRResNet
    m = NPRResNet(num_classes=1)
    raw = torch.load(f"{BASE}/checkpoints/NPR.pth", map_location='cpu')
    st = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
    st = {k.replace('module.','',1): v for k,v in st.items()}
    m.load_state_dict(st, strict=True); return m.to(dev).eval()

def load_ff():
    from models import build_model
    args = argparse.Namespace(backbone='CLIP:ViT-L/14', num_vit_adapter=3, num_context_embedding=8,
        init_context_embedding=None, num_classes=2)
    m = build_model(args)
    ck = torch.load(f"{BASE}/checkpoints/fatformer.pth", map_location='cpu')
    m.load_state_dict(ck['model'] if 'model' in ck else ck, strict=False); return m.to(dev).eval()

def score_paths(model, tf, kind, paths):
    out = []
    for p in paths[:CAP]:
        pil = norm_pil(p)
        if pil is None: continue
        with torch.no_grad():
            o = model(tf(pil).unsqueeze(0).to(dev).float())
            out.append(torch.sigmoid(o).item() if kind=='npr' else o.softmax(1)[0,1].item())
    return np.array(out)

def need():  # list of (method, fam, cls) not yet cached
    todo=[]
    for meth in ["npr","fatformer"]:
        for fam in FAM:
            for cls in ["nature","ai"]:
                if not os.path.exists(f"{CDIR}/{meth}_{fam.replace(' ','')}_{cls}.npy"): todo.append((meth,fam,cls))
    return todo

todo = need()
print(f"device={dev}  todo={len(todo)} sets", flush=True)
model_cache={}
for meth, fam, cls in todo:
    if meth not in model_cache:
        print(f"loading {meth}...", flush=True)
        model_cache[meth] = (load_npr(),'npr',npr_tf) if meth=="npr" else (load_ff(),'ff',ff_tf)
    m,kind,tf = model_cache[meth]
    t=time.time(); sc = score_paths(m, tf, 'npr' if meth=='npr' else 'ff', gf(fam,cls))
    np.save(f"{CDIR}/{meth}_{fam.replace(' ','')}_{cls}.npy", sc)
    print(f"  {meth} {fam} {cls}: n={len(sc)} in {time.time()-t:.0f}s", flush=True)

# ---- all cached? compute table ----
if need():
    print("still incomplete; re-run to resume", flush=True); sys.exit(0)
def ld(meth,fam,cls): return np.load(f"{CDIR}/{meth}_{fam.replace(' ','')}_{cls}.npy")
lines=["DEEP BASELINES (zero-shot, MPS) on corrected content-matched real-vs-AI (native256 Q95 444, best-dir AUC)",
       f"{'test set':13s} {'NPR':>7s} {'FatFormer':>10s}"]
npr_m,ff_m=[],[]
for fam in FAM:
    aucs={}
    for meth in ["npr","fatformer"]:
        r=ld(meth,fam,"nature"); a=ld(meth,fam,"ai")
        y=np.r_[np.zeros(len(r)),np.ones(len(a))]; s=np.r_[r,a]
        au=roc_auc_score(y,s); aucs[meth]=max(au,1-au)
    lines.append(f"{fam:13s} {aucs['npr']:7.3f} {aucs['fatformer']:10.3f}")
    if fam!="SD v1.5": npr_m.append(aucs['npr']); ff_m.append(aucs['fatformer'])
lines.append(f"{'mean(5 unseen)':13s} {np.mean(npr_m):7.3f} {np.mean(ff_m):10.3f}")
lines.append("")
lines.append("OUR methods (multi-gen leave-one-out): OPF 0.734  Base 0.887  Comb 0.900  (cross mean)")
txt="\n".join(lines); print("\n"+txt, flush=True)
open(f"{SC}/deep_compare_results.txt","w").write(txt+"\n")
print("WROTE deep_compare_results.txt", flush=True)
