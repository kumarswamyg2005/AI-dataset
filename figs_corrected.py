"""
Regenerate the two paper figures on GENUINE data:
  fig_method.pdf  — (a) DCT band partition, (b) real(ImageNet nature) vs AI(GenImage) LF-HF coupling
                    scatter, using representative (median-coupling) images (not cherry-picked).
  fig_results.pdf — grouped bar chart: our Fusion vs prior NPR/FatFormer across the 5 unseen
                    generators + mean. Colorblind-safe (Okabe-Ito), print-friendly.
"""
import sys, glob, numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, "/Users/kumaraswamy/Desktop/jpg/research/full")
import opf_local_train as L
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.constrained_layout.use": True})
TG = "/Users/kumaraswamy/Desktop/jpg/research/data/tiny_genimage"
# Okabe-Ito colorblind-safe palette
BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#999999"

def y_bands(path):
    bgr = cv2.imread(path)
    if bgr is None: return None
    bgr = L._center_crop(bgr, L.CONFIG.CENTER_CROP)
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)[:, :, 0]
    eL, eH = L._band_energies(y, [L.LF_MASK, L.HF_MASK])
    return np.asarray(eL), np.asarray(eH)
def rho(path):
    b = y_bands(path)
    if b is None or len(b[0]) < 8: return None
    return L._pearson_safe(b[0], b[1])
def pick_median(paths, n=120):
    vals = []
    for p in paths[:n]:
        r = rho(p)
        if r is not None and np.isfinite(r): vals.append((r, p))
    if not vals: return None, None
    rs = np.array([v[0] for v in vals]); i = int(np.argmin(np.abs(rs - np.median(rs))))
    return vals[i][1], float(np.median(rs))

# genuine content-matched examples (SD v1.5 family)
real_img, real_med = pick_median(sorted(glob.glob(f"{TG}/imagenet_ai_0424_sdv5/train/nature/*")))
ai_img,   ai_med   = pick_median(sorted(glob.glob(f"{TG}/imagenet_ai_0424_sdv5/train/ai/*")))
print(f"real median rho={real_med:.3f} -> {real_img}")
print(f"AI   median rho={ai_med:.3f} -> {ai_img}")

# ---------------- Figure 1 ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
u, v = np.meshgrid(np.arange(8), np.arange(8), indexing="ij"); d = u + v
band = np.zeros((8, 8)); band[(d <= 2) & (d > 0)] = 1; band[(d >= 3) & (d <= 4)] = 2; band[(d > 4)] = 3
cmap = matplotlib.colors.ListedColormap(["#1f2937", "#7FB8E6", "#34d399", "#f59e0b"])
ax[0].imshow(band, cmap=cmap, vmin=0, vmax=3)
ax[0].set_xticks(range(8)); ax[0].set_yticks(range(8))
ax[0].set_xlabel("horizontal freq $u$"); ax[0].set_ylabel("vertical freq $v$")
ax[0].set_title("(a) $8{\\times}8$ DCT band partition"); ax[0].grid(False)
ax[0].legend(handles=[Patch(color="#1f2937", label="DC (excl.)"), Patch(color="#7FB8E6", label="VLF (5)"),
    Patch(color="#34d399", label="MF (9)"), Patch(color="#f59e0b", label="HF (49)")],
    fontsize=6.5, loc="upper right", framealpha=0.9)
rL, rH = y_bands(real_img); aL, aH = y_bands(ai_img)
lz = lambda x: np.log10(x + 1.0)
rr = L._pearson_safe(rL, rH); ar = L._pearson_safe(aL, aH)
ax[1].scatter(lz(aL), lz(aH), s=3, alpha=0.35, color=VERM, label=f"AI    ($\\rho{{=}}{ar:.2f}$)")
ax[1].scatter(lz(rL), lz(rH), s=3, alpha=0.35, color=BLUE, label=f"real  ($\\rho{{=}}{rr:.2f}$)")
ax[1].set_xlabel("$\\log_{10} e_L^{Y}$  (low-band energy)")
ax[1].set_ylabel("$\\log_{10} e_H^{Y}$  (high-band energy)")
ax[1].set_title("(b) Per-block low-to-high coupling"); ax[1].legend(fontsize=7.5, loc="upper left")
fig.savefig("figs/fig_method.png", dpi=300, bbox_inches="tight")
print(f"wrote figs/fig_method.png (real rho={rr:.3f}, AI rho={ar:.3f})")

# ---------------- Figure 2: results bar chart ----------------
gens = ["ADM", "GLIDE", "Midjrn.", "Wukong", "VQDM", "Mean"]
# regenerated artifacts: table1_final_results.txt + deep_full_results.txt (2026-07-05)
fusion = [0.882, 0.930, 0.856, 0.938, 0.916, 0.904]
fatf   = [0.961, 0.908, 0.660, 0.904, 0.967, 0.880]
npr    = [0.653, 0.624, 0.590, 0.507, 0.553, 0.585]
x = np.arange(len(gens)); w = 0.26
fig2, ax2 = plt.subplots(figsize=(7.2, 2.7))
ax2.bar(x - w, fusion, w, color=BLUE,  label="Fusion (ours)")
ax2.bar(x,     fatf,   w, color=VERM,  label="FatFormer$^\\ddagger$")
ax2.bar(x + w, npr,    w, color=GREEN, label="NPR$^\\ddagger$")
ax2.axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.6)
ax2.text(len(gens)-0.5, 0.505, "chance", fontsize=6.5, va="bottom", ha="right", alpha=0.7)
ax2.set_ylim(0.5, 1.0); ax2.set_xticks(x); ax2.set_xticklabels(gens)
ax2.set_ylabel("AUC"); ax2.set_axisbelow(True)
ax2.axvline(len(gens)-1.5, color="k", lw=0.6, alpha=0.25)  # separate Mean
ax2.set_title("Our fusion vs. zero-shot deep detectors (unseen generators)")
ax2.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
fig2.savefig("figs/fig_results.png", dpi=300, bbox_inches="tight")
print("wrote figs/fig_results.png")
