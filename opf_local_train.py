#!/usr/bin/env python
"""
OPF local terminal trainer — full scale-up on this Mac, no Kaggle 9-hour limit.
Streams AI images from HuggingFace (authenticated + parallel sharded) and
extracts features from locally downloaded real/AI datasets.

GENERATED from opf_kaggle_training.py by scratchpad/build_local.py. The feature
extractor, checkpoint logic, PATH A multiprocessing, and training are copied
VERBATIM (validated by the 64-check harness). Local additions: paths, HF-token
auth, parallel sharded streaming (run_hf_streaming_parallel), and run phases.

Usage:
  python opf_local_train.py --phase stream   # PATH B only (HF streaming)
  python opf_local_train.py --phase files    # PATH A only (local datasets)
  python opf_local_train.py --phase train    # train on whatever is extracted
  python opf_local_train.py --phase all      # files -> stream -> train (default)
"""
import os, sys, json, time, math, pickle, logging, argparse, warnings
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image as PILImage
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")
import random
random.seed(42); np.random.seed(42)

logger = logging.getLogger("opf_local")
logger.setLevel(logging.INFO)

class CONFIG:
    ROOT             = Path("/Users/kumaraswamy/Desktop/jpg/research")
    DATA_DIR         = ROOT / "data"
    # OPF_WORKING_DIR lets a test/smoke run redirect outputs (inherited by the
    # spawned streaming workers via the environment).
    WORKING_DIR      = Path(os.environ.get("OPF_WORKING_DIR",
                                           str(ROOT / "full" / "local_run")))
    CHECKPOINT_DIR   = WORKING_DIR / "checkpoint"
    LOG_PATH         = WORKING_DIR / "training.log"
    SKIPPED_LOG_PATH = WORKING_DIR / "skipped_files.log"

    # PATH A — local datasets, finalized after inspecting the downloads so we
    # never mislabel real images as AI:
    #   * SIDD has GT (clean) + NOISY pairs per scene -> keep GT only.
    #   * ai-vs-human mixes real/AI in one folder -> select AI via its CSV.
    #   * tiny-genimage (GenImage) is DROPPED: it is the same corpus as the
    #     paper's AI holdout (GenImage[3000:3999]) and the Kaggle copy's
    #     ordering can't be mapped to guarantee disjointness -> exclude to keep
    #     the holdout leakage-free. (We have ~530k AI without it.)
    DATA_SOURCES = [
        # real camera images (label 0)
        {"name": "dresden", "label": 0, "path": str(DATA_DIR / "dresden"),
         "start_index": 0, "max_count": 20000},
        {"name": "sidd_gt", "label": 0, "path": str(DATA_DIR / "sidd"),
         "include_substr": "GT_SRGB", "start_index": 0, "max_count": 5000},
        # AI images (label 1) — select only AI rows (label==1) from the CSV
        {"name": "ai_vs_human", "label": 1, "path": str(DATA_DIR / "ai_vs_human"),
         "csv": {"file": "train.csv", "name_col": "file_name",
                 "label_col": "label", "ai_value": "1"},
         "start_index": 0, "max_count": 40000},
    ]

    # PATH B — HuggingFace streaming (label-filtered to AI only).
    HF_SOURCES = [
        {"name": "openfake", "repo_id": "ComplexDataLab/OpenFake", "config": "core",
         "split": "train", "image_field": "image", "label": 1,
         "label_field": "label", "ai_values": ["fake"], "max_count": 350000},
        {"name": "midjourney", "repo_id": "ehristoforu/midjourney-images", "config": None,
         "split": "train", "image_field": "image", "label": 1,
         "label_field": None, "ai_values": None, "max_count": 50000},
        {"name": "defactify_cocoai", "repo_id": "Rajarshi-Roy-research/Defactify_Image_Dataset",
         "config": None, "split": "train", "image_field": "Image", "label": 1,
         "label_field": "Label_A", "ai_values": [1], "max_count": 90000},
    ]

    HOLDOUT_FORBIDDEN_SUBSTRINGS = ["raise_holdout", "raise-1k", "raise_1k",
                                    "holdout_leakage_free", "holdout_final_1k"]
    GENIMAGE_HOLDOUT_START = 3000
    GENIMAGE_HOLDOUT_END   = 4000

    MIN_IMAGE_DIM    = 256
    CENTER_CROP      = 512
    BATCH_SIZE       = 500    # smaller -> first real checkpoint lands sooner
    N_WORKERS        = 4    # PATH A (CPU) — gentle: half of 8 cores
    N_STREAM_WORKERS = 4    # PATH B — measured best here. 8 was ~5x SLOWER:
                            # HF throttles per-token concurrency and OpenFake's
                            # ~5GB shards thrash when 8 streams contend. 4 wins.

    # No 9-hour limit locally: make the budget effectively infinite so the
    # verbatim run_file_extraction never early-exits.
    SESSION_BUDGET_S = 10**12
    SAFETY_MARGIN_S  = 0

    N_ESTIMATORS = 500
    N_SPLITS     = 5
    SEED         = 42

    FINAL_MODEL_PATH = WORKING_DIR / "opf_final_model.pkl"
    BEST_FOLD_PATH   = WORKING_DIR / "opf_best_fold.pkl"
    SCALER_PATH      = WORKING_DIR / "opf_scaler.pkl"
    REPORT_PATH      = WORKING_DIR / "training_report.json"


BLOCK_SIZE    = 8
VLF_THRESHOLD = 2
LF_THRESHOLD  = 4
MI_BINS       = 20

FEATURE_NAMES = [
    "lf_hf_pearson_y",  "lf_hf_pearson_cb", "lf_hf_pearson_cr",
    "mi_mean_y",        "mi_mean_cr",
    "ratio_mean_cb",    "ratio_var_cb",     "ratio_mean_cr",    "ratio_var_cr",
    "pearson_mf_hf_cr", "pearson_vlf_mf_cr",
    "pearson_cr_cb_lf", "pearson_cr_cb_hf",
    "pearson_y_cr_lf",  "pearson_y_cr_hf",
]
assert len(FEATURE_NAMES) == 15
N_FEATURES = 15


def _build_band_masks():
    """
    Boolean 8×8 masks for VLF / MF / LF / HF DCT bands.

    FORENSIC RATIONALE:
        The camera lens MTF is a spatial low-pass filter. Within each 8×8 JPEG
        block, this creates a physical coupling between LF and HF band energy.
        Splitting on the u+v diagonal lets us measure that coupling directly.
        DC=(0,0) carries only block brightness and is excluded everywhere.
    """
    u, v = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    d    = u + v
    dc   = (u == 0) & (v == 0)
    vlf  = (d <= VLF_THRESHOLD) & ~dc
    mf   = (d > VLF_THRESHOLD) & (d <= LF_THRESHOLD) & ~dc
    lf   = (d <= LF_THRESHOLD) & ~dc
    hf   = (d > LF_THRESHOLD)  & ~dc
    assert vlf.sum() == 5 and mf.sum() == 9 and lf.sum() == 14 and hf.sum() == 49
    return vlf, mf, lf, hf

VLF_MASK, MF_MASK, LF_MASK, HF_MASK = _build_band_masks()


def _pearson_safe(x, y):
    """
    Pearson r, returning 0.0 for degenerate (constant / too-short) inputs.

    FORENSIC RATIONALE:
        Flat channels (saturated skies, uniform backgrounds) have zero
        variance and undefined Pearson r — returning 0 prevents NaN injection.
    """
    if len(x) < 4 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    r, _ = stats.pearsonr(x, y)
    return float(r) if np.isfinite(r) else 0.0


def _mutual_information(x, y):
    """
    Histogram MI estimate (20×20 bins, 1/99-percentile clipped).

    FORENSIC RATIONALE:
        Camera optics induce nonlinear LF-HF dependence that Pearson alone
        misses. MI captures any statistical coupling, linear or not.
    """
    if len(x) < 4:
        return 0.0
    xc = np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
    yc = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))
    h, _, _ = np.histogram2d(xc, yc, bins=MI_BINS)
    p = h / (h.sum() + 1e-12)
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    eps = 1e-12
    return float(max(np.sum(p * np.log((p + eps) / (px * py + eps))), 0.0))


def _band_energies(plane, masks):
    """
    Per-block L2 energy for each DCT band mask on a single channel plane.

    FORENSIC RATIONALE:
        By Parseval's theorem, sum(coeff²) in a band equals the spatial-domain
        energy of that frequency band within the block. The vector of per-block
        energies is the spatial map whose cross-band correlations carry the
        camera-optics fingerprint.
    """
    H, W = plane.shape
    nh, nw = H // BLOCK_SIZE, W // BLOCK_SIZE
    n = nh * nw
    if n < 4:
        return [np.zeros(0) for _ in masks]
    out = [np.empty(n, dtype=np.float64) for _ in masks]
    b = 0
    for r in range(nh):
        r0 = r * BLOCK_SIZE
        for c in range(nw):
            c0 = c * BLOCK_SIZE
            blk = plane[r0:r0+BLOCK_SIZE, c0:c0+BLOCK_SIZE].astype(np.float32)
            d = cv2.dct(blk)
            for k, m in enumerate(masks):
                out[k][b] = float(np.sum(d[m] ** 2))
            b += 1
    return out


def _center_crop(bgr, size):
    """Center-crop to size×size; return as-is if smaller."""
    H, W = bgr.shape[:2]
    if H < size or W < size:
        return bgr
    return bgr[(H-size)//2:(H-size)//2+size, (W-size)//2:(W-size)//2+size]


def _extract_from_bgr(bgr):
    """
    Core 15-feature extractor operating on a BGR numpy array.

    FORENSIC RATIONALE:
        All 15 features live in the JPEG DCT domain (Groups 1-4). The camera
        optical pipeline (lens MTF + Bayer CFA demosaicing) imprints a specific
        LF-HF coupling in the Cr channel that AI generators cannot replicate.
        Processing in YCrCb ensures Cr/Cb are the actual demosaiced chrominance
        channels, not post-hoc colour transforms.

    Returns float32 array of shape (15,), or None if the image is unusable.
    """
    if bgr is None:
        return None
    H, W = bgr.shape[:2]
    if H < CONFIG.MIN_IMAGE_DIM or W < CONFIG.MIN_IMAGE_DIM:
        return None

    bgr = _center_crop(bgr, CONFIG.CENTER_CROP)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    planes = {"y": ycrcb[:,:,0], "cb": ycrcb[:,:,2], "cr": ycrcb[:,:,1]}

    E = {ch: _band_energies(p, [VLF_MASK, MF_MASK, LF_MASK, HF_MASK])
         for ch, p in planes.items()}

    if len(E["y"][0]) < 4:
        return None

    y_vlf,  y_mf,  y_lf,  y_hf  = E["y"]
    cb_vlf, cb_mf, cb_lf, cb_hf = E["cb"]
    cr_vlf, cr_mf, cr_lf, cr_hf = E["cr"]
    eps = 1e-6

    # Group 1 — MTF LF-HF coupling
    g1 = [_pearson_safe(y_lf,  y_hf),
          _pearson_safe(cb_lf, cb_hf),
          _pearson_safe(cr_lf, cr_hf),
          _mutual_information(y_lf,  y_hf),
          _mutual_information(cr_lf, cr_hf)]

    # Group 2 — HF/LF energy ratios
    rcb = cb_hf / (cb_lf + eps)
    rcr = cr_hf / (cr_lf + eps)
    g2 = [float(np.mean(rcb)), float(np.var(rcb)),
          float(np.mean(rcr)), float(np.var(rcr))]

    # Group 3 — Cr multi-band coupling
    g3 = [_pearson_safe(cr_mf, cr_hf),
          _pearson_safe(cr_vlf, cr_mf)]

    # Group 4 — cross-channel coupling
    g4 = [_pearson_safe(cr_lf, cb_lf),
          _pearson_safe(cr_hf, cb_hf),
          _pearson_safe(y_lf,  cr_lf),
          _pearson_safe(y_hf,  cr_hf)]

    fv = np.array(g1 + g2 + g3 + g4, dtype=np.float32)
    if fv.shape[0] != N_FEATURES or not np.all(np.isfinite(fv)):
        return None
    return fv


def extract_opf_features(image_path):
    """Load a JPEG from disk and call _extract_from_bgr."""
    return _extract_from_bgr(cv2.imread(str(image_path)))


def extract_opf_features_from_pil(pil_img):
    """
    Convert a PIL Image to BGR and call _extract_from_bgr.
    Used by the HuggingFace streaming path (Path B).
    Handles RGBA, palette mode, and greyscale gracefully.
    """
    try:
        rgb = pil_img.convert("RGB")
        bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        return _extract_from_bgr(bgr)
    except Exception:
        return None


MANIFEST_PATH = CONFIG.CHECKPOINT_DIR / "manifest.json"


# ── shared helpers ────────────────────────────────────────────────────────────

def _is_holdout_path(p):
    low = p.lower()
    return any(s in low for s in CONFIG.HOLDOUT_FORBIDDEN_SUBSTRINGS)


def _load_manifest():
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH) as f:
                m = json.load(f)
            # ensure hf_sources key exists for old manifests
            m.setdefault("hf_sources", {})
            logger.info("Resumed manifest: %d file-based + hf_sources=%s",
                        len(m["processed"]), list(m["hf_sources"]))
            return m
        except Exception as e:
            logger.warning("Manifest unreadable (%s); starting fresh.", e)
    return {"processed": [], "n_batches": 0, "hf_sources": {}}


def _save_manifest(manifest):
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, MANIFEST_PATH)


def _save_batch(idx, feats, labels, paths):
    np.savez_compressed(
        str(CONFIG.CHECKPOINT_DIR / f"batch_{idx:05d}.npz"),
        features=np.asarray(feats, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        paths=np.asarray(paths),
        feature_names=np.asarray(FEATURE_NAMES),
    )


# ── PATH A: file-based extraction ─────────────────────────────────────────────

def _collect_source_images(source):
    root  = Path(source["path"])
    label = source["label"]
    name  = source["name"]
    if not root.exists():
        logger.warning("Source '%s' path not found: %s (skip)", name, root)
        return []
    exts = {".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG",".BMP"}
    all_paths = sorted(str(p) for p in root.rglob("*")
                       if p.suffix in exts and p.is_file())
    for p in all_paths:
        assert not _is_holdout_path(p), \
            f"LEAKAGE GUARD: holdout image in training path: {p}"
    start = source.get("start_index", 0)
    is_gi = "genimage" in name.lower() or "genimage" in str(root).lower()
    if is_gi:
        all_paths = [p for i, p in enumerate(all_paths)
                     if not (CONFIG.GENIMAGE_HOLDOUT_START
                             <= i < CONFIG.GENIMAGE_HOLDOUT_END)]
        assert start >= CONFIG.GENIMAGE_HOLDOUT_END, \
            f"GenImage start_index {start} must be >= {CONFIG.GENIMAGE_HOLDOUT_END}"
    sliced = all_paths[start:]
    mc = source.get("max_count")
    if mc:
        sliced = sliced[:mc]
    logger.info("Source '%s' label=%d: %d images", name, label, len(sliced))
    return [(p, label) for p in sliced]


def _worker(item):
    path, label = item
    try:
        fv = extract_opf_features(path)
        return ("OK", label, path, fv) if fv is not None \
               else ("SKIP", "unusable", path)
    except Exception as e:
        return ("SKIP", f"exc:{type(e).__name__}", path)


def run_file_extraction(manifest, t_start):
    """
    PATH A: extract features from all Kaggle-attached datasets.
    Returns (manifest, complete) where complete=False means time ran out.
    """
    work = []
    for src in CONFIG.DATA_SOURCES:
        work.extend(_collect_source_images(src))
    logger.info("PATH A total work items: %d", len(work))

    done = set(manifest["processed"])
    pending = [(p, l) for p, l in work if p not in done]
    logger.info("PATH A pending (after resume): %d", len(pending))
    if not pending:
        logger.info("PATH A already complete.")
        return manifest, True

    n_workers = CONFIG.N_WORKERS or max(1, (os.cpu_count() or 4) - 1)
    logger.info("PATH A: %d workers, batch_size=%d", n_workers, CONFIG.BATCH_SIZE)

    batch_idx  = manifest["n_batches"]
    buf_f, buf_l, buf_p = [], [], []
    n_skip = 0

    def _flush():
        nonlocal batch_idx, buf_f, buf_l, buf_p
        if not buf_f:
            return
        _save_batch(batch_idx, buf_f, buf_l, buf_p)
        manifest["processed"].extend(buf_p)
        manifest["n_batches"] = batch_idx + 1
        _save_manifest(manifest)
        logger.info("PATH A checkpoint: batch %05d (%d imgs), total=%d",
                    batch_idx, len(buf_f), len(manifest["processed"]))
        batch_idx += 1
        buf_f, buf_l, buf_p = [], [], []

    with mp.Pool(processes=n_workers) as pool:
        with tqdm(total=len(pending), desc="PATH A (files)", unit="img") as pb:
            for res in pool.imap_unordered(_worker, pending, chunksize=8):
                if res[0] == "OK":
                    _, lbl, pth, fv = res
                    buf_f.append(fv); buf_l.append(lbl); buf_p.append(pth)
                else:
                    _, reason, pth = res; n_skip += 1
                    with open(CONFIG.SKIPPED_LOG_PATH, "a") as sf:
                        sf.write(f"A\t{reason}\t{pth}\n")
                pb.update(1)
                pb.set_postfix_str(f"ok={len(buf_f)} skip={n_skip}")
                if len(buf_f) >= CONFIG.BATCH_SIZE:
                    _flush()
                    if (CONFIG.SESSION_BUDGET_S - (time.time()-t_start)
                            < CONFIG.SAFETY_MARGIN_S):
                        logger.warning("PATH A: time budget low. Checkpointing.")
                        pool.terminate()
                        _flush()
                        return manifest, False

    _flush()
    logger.info("PATH A complete. Skipped %d images.", n_skip)
    return manifest, True


# ── PATH B: HuggingFace streaming extraction ──────────────────────────────────

def _get_pil_from_item(item, image_field):
    """
    Retrieve a PIL Image from a HuggingFace dataset item.

    Matches the image key CASE-INSENSITIVELY (e.g. Defactify uses "Image",
    not "image") and tries common fallbacks. Handles four storage forms:
    a decoded PIL Image, raw bytes, the HF Image-feature dict with inline
    {"bytes": ...}, or that dict with a {"path": "hf://..."} reference (some
    datasets, e.g. ehristoforu/midjourney-images, store the image by path with
    bytes=None — resolved here via fsspec using the saved HF token).
    """
    import io
    lower_map = {k.lower(): k for k in item.keys()}
    for key in [image_field, "image", "img", "pixel_values", "jpg", "png"]:
        actual = lower_map.get(key.lower())
        if actual is None:
            continue
        val = item[actual]
        if isinstance(val, PILImage.Image):
            return val
        if isinstance(val, (bytes, bytearray)):
            try:
                return PILImage.open(io.BytesIO(val))
            except Exception:
                continue
        if isinstance(val, dict):
            if val.get("bytes"):
                try:
                    return PILImage.open(io.BytesIO(val["bytes"]))
                except Exception:
                    continue
            if val.get("path"):
                try:
                    import fsspec
                    with fsspec.open(val["path"], "rb") as fh:
                        return PILImage.open(io.BytesIO(fh.read()))
                except Exception:
                    continue
    return None


def _hf_item_is_ai(item, src):
    """
    Decide whether a streamed HF item is an AI image we should keep.

    If the source has no `label_field`, every item is the configured label
    (whole-dataset AI). Otherwise read that field case-insensitively and keep
    the item only when its value is in `ai_values` — this prevents the REAL
    photos in real-vs-fake benchmarks (OpenFake, Defactify) from being
    mislabelled as AI and poisoning training.
    """
    lf = src.get("label_field")
    if not lf:
        return True
    lower_map = {k.lower(): k for k in item.keys()}
    actual = lower_map.get(lf.lower())
    if actual is None:
        return True   # field missing -> fall back to configured label
    val = item[actual]
    if isinstance(val, str):
        val = val.strip().lower()
    ai_values = src.get("ai_values") or []
    norm = {v.lower() if isinstance(v, str) else v for v in ai_values}
    return val in norm

# ============================================================================
#  OVERRIDE — local _collect_source_images: adds CSV-based AI selection and an
#  include-substring filter, so AI sources never pick up real images. Same
#  leakage guards as the verbatim version (RAISE substrings + GenImage window).
#  Defined after the verbatim copy so this definition wins at call time.
# ============================================================================
def _collect_source_images(source):
    root  = Path(source["path"])
    label = source["label"]
    name  = source["name"]
    if not root.exists():
        logger.warning("Source '%s' path not found: %s (skip)", name, root)
        return []

    csv_cfg = source.get("csv")
    if csv_cfg:
        import csv as _csv
        ai_val = str(csv_cfg.get("ai_value", "1"))
        all_paths = []
        with open(root / csv_cfg["file"]) as f:
            for row in _csv.DictReader(f):
                if str(row.get(csv_cfg["label_col"], "")).strip() == ai_val:
                    all_paths.append(str(root / row[csv_cfg["name_col"]]))
        all_paths.sort()
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
        all_paths = sorted(str(p) for p in root.rglob("*")
                           if p.suffix in exts and p.is_file())

    inc = source.get("include_substr")
    if inc:
        all_paths = [p for p in all_paths if inc in p]

    for p in all_paths:
        assert not _is_holdout_path(p), \
            f"LEAKAGE GUARD: holdout image in training path: {p}"

    start = source.get("start_index", 0)
    is_gi = "genimage" in name.lower() or "genimage" in str(root).lower()
    if is_gi:
        all_paths = [p for i, p in enumerate(all_paths)
                     if not (CONFIG.GENIMAGE_HOLDOUT_START
                             <= i < CONFIG.GENIMAGE_HOLDOUT_END)]
        assert start >= CONFIG.GENIMAGE_HOLDOUT_END, \
            f"GenImage start_index {start} must be >= {CONFIG.GENIMAGE_HOLDOUT_END}"

    sliced = all_paths[start:]
    mc = source.get("max_count")
    if mc:
        sliced = sliced[:mc]
    logger.info("Source '%s' label=%d: %d images", name, label, len(sliced))
    return [(p, label) for p in sliced]


# ============================================================================
#  PATH B (LOCAL) — parallel sharded HuggingFace streaming
#  Each source is split across N_STREAM_WORKERS processes via
#  IterableDataset.shard(); every worker reuses the SAME per-image logic
#  (_hf_item_is_ai filter, _get_pil_from_item, extract_opf_features_from_pil,
#  decode=False, defensive next) and checkpoints independently so the whole
#  thing is resumable with no duplication.
# ============================================================================

def _hf_state_path(name, w):
    return CONFIG.CHECKPOINT_DIR / f"hfstate_{name}_w{w}.json"

def _hf_read_states(name, W):
    kept = consumed = 0
    for w in range(W):
        sp = _hf_state_path(name, w)
        if sp.exists():
            try:
                s = json.load(open(sp))
                kept += int(s.get("count", 0))
                consumed += int(s.get("scan_pos", s.get("stream_pos", 0)))
            except Exception:
                pass
    return kept, consumed

def _hf_done_count(name, W):
    return _hf_read_states(name, W)[0]

def _stream_worker(src, w, W, target):
    """One shard of one HF source. Runs in its own process (spawn-safe)."""
    from datasets import load_dataset
    try:
        from datasets import Image as HFImage
    except Exception:
        HFImage = None

    name = src["name"]
    ckpt = CONFIG.CHECKPOINT_DIR
    ckpt.mkdir(parents=True, exist_ok=True)
    state_path = _hf_state_path(name, w)

    kept, stream_pos, kb = 0, 0, 0
    if state_path.exists():
        try:
            s = json.load(open(state_path))
            kept, stream_pos, kb = s["count"], s["stream_pos"], s.get("batches", 0)
        except Exception:
            pass
    if kept >= target:
        return (name, w, kept, stream_pos)

    try:
        ds = load_dataset(src["repo_id"], src["config"],
                          split=src["split"], streaming=True)
        ds = ds.shard(num_shards=W, index=w)
        if HFImage is not None and src.get("image_field"):
            try:
                ds = ds.cast_column(src["image_field"], HFImage(decode=False))
            except Exception:
                pass
        if stream_pos > 0:
            ds = ds.skip(stream_pos)
    except Exception as e:
        print(f"[{name} w{w}] load failed: {e}", flush=True)
        return (name, w, kept, stream_pos)

    buf_f, buf_l, buf_p = [], [], []
    n_skip = 0
    n_filt = 0
    # Resume-safe checkpoint (count + stream_pos) advances only at a batch flush.
    # scan_pos is a separate liveness heartbeat = current consumed position,
    # written every ~15s so the parent's stall watchdog sees genuine progress
    # long before the first flush (critical with many workers / large shards).
    safe_count, safe_pos = kept, stream_pos
    last_live = time.time()

    def _write_state():
        tmp = state_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"count": safe_count, "stream_pos": safe_pos,
                       "scan_pos": stream_pos, "batches": kb}, f)
        os.replace(tmp, state_path)

    def flush():
        nonlocal kb, buf_f, buf_l, buf_p, safe_count, safe_pos
        if buf_f:
            np.savez_compressed(
                str(ckpt / f"batch_{name}_w{w}_{kb:05d}.npz"),
                features=np.asarray(buf_f, dtype=np.float32),
                labels=np.asarray(buf_l, dtype=np.int8),
                paths=np.asarray(buf_p),
                feature_names=np.asarray(FEATURE_NAMES),
            )
            kb += 1
            buf_f, buf_l, buf_p = [], [], []
        safe_count, safe_pos = kept, stream_pos   # resume-safe point
        _write_state()

    consec = 0
    it = iter(ds)
    while kept < target:
        try:
            item = next(it)
        except StopIteration:
            break
        except Exception:
            stream_pos += 1; n_skip += 1; consec += 1
            if consec >= 50:
                print(f"[{name} w{w}] 50 consec read errors near pos {stream_pos}; stop",
                      flush=True)
                break
            continue
        consec = 0
        stream_pos += 1
        if time.time() - last_live > 15:    # liveness heartbeat (not a flush)
            _write_state(); last_live = time.time()
        if not _hf_item_is_ai(item, src):
            n_filt += 1
            continue
        try:
            pil = _get_pil_from_item(item, src["image_field"])
            fv = extract_opf_features_from_pil(pil) if pil is not None else None
        except Exception:
            fv = None
        if fv is not None:
            buf_f.append(fv); buf_l.append(src["label"])
            buf_p.append(f"{name}_w{w}_{stream_pos}")
            kept += 1
            if len(buf_f) >= CONFIG.BATCH_SIZE:
                flush()
        else:
            n_skip += 1
    flush()   # writes any remaining batch AND persists final stream_pos
    print(f"[{name} w{w}] done: kept={kept} pos={stream_pos} filt={n_filt} skip={n_skip}",
          flush=True)
    # HuggingFace streaming (datasets/fsspec) leaves non-daemon threads that can
    # block normal process teardown, hanging the parent's join() forever. All
    # state is already on disk and the return value is unused by the parent
    # (it reads hfstate files), so a spawned worker exits hard. The in-process
    # unit test (main process) returns normally.
    sys.stdout.flush()
    if mp.parent_process() is not None:
        os._exit(0)
    return (name, w, kept, stream_pos)

def run_hf_streaming_parallel():
    from datasets import load_dataset
    ctx = mp.get_context("spawn")
    cap = int(os.environ.get("OPF_MAX_PER_SOURCE", "0") or "0")  # 0 = no cap
    for src in CONFIG.HF_SOURCES:
        name, max_count = src["name"], src["max_count"]
        if cap:
            max_count = min(max_count, cap)
        # Match the worker count to the number of shards. Contiguous sharding
        # gives empty shards to extra workers when num_shards < N_STREAM_WORKERS
        # (MidJourney has 1 shard), which would leave most workers idle AND cap
        # delivery at one worker's quota. Probing num_shards is metadata-only.
        try:
            _probe = load_dataset(src["repo_id"], src["config"],
                                  split=src["split"], streaming=True)
            nshards = int(_probe.num_shards)
            del _probe
        except Exception as e:
            logger.warning("PATH B '%s': shard probe failed (%s); assuming 1.", name, e)
            nshards = 1
        W = max(1, min(CONFIG.N_STREAM_WORKERS, nshards))
        done = _hf_done_count(name, W)
        if done >= max_count:
            logger.info("PATH B '%s': already complete (%d/%d).", name, done, max_count)
            continue
        target_per = math.ceil(max_count / W)
        logger.info("PATH B '%s': %d shards -> %d workers x %d target (max %d), resuming from %d.",
                    name, nshards, W, target_per, max_count, done)
        procs = [ctx.Process(target=_stream_worker, args=(src, w, W, target_per))
                 for w in range(W)]
        for p in procs:
            p.start()
        STALL_S = 1800   # 30 min with zero rows consumed -> assume a network stall
                         # (workers heartbeat scan_pos every ~15s, so a true
                         #  stall is unambiguous; generous to ride out slow shards)
        _, last_consumed = _hf_read_states(name, W)
        last_progress_t = time.time()
        while any(p.is_alive() for p in procs):
            time.sleep(30)
            kept, consumed = _hf_read_states(name, W)
            logger.info("PATH B '%s' progress: kept=%d / %d (consumed=%d)",
                        name, kept, max_count, consumed)
            if consumed > last_consumed:
                last_consumed = consumed
                last_progress_t = time.time()
            elif time.time() - last_progress_t > STALL_S:
                logger.warning("PATH B '%s': no progress for %d min — terminating "
                               "workers; re-run to resume.", name, STALL_S // 60)
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                break
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        logger.info("PATH B '%s' done: kept=%d", name, _hf_done_count(name, W))
    logger.info("PATH B complete.")


# ============================================================================
#  Training (cells 8-14 from the notebook, wrapped so it runs only on demand)
# ============================================================================
def run_training():
    # ============================================================================
    # In[8]:  Load & merge all feature batches
    # ============================================================================
    def load_all_batches():
        batch_files = sorted(CONFIG.CHECKPOINT_DIR.glob("batch_*.npz"))
        if not batch_files:
            raise RuntimeError("No feature batches found in checkpoint/.")
        Xs, ys, ps = [], [], []
        for bf in batch_files:
            d = np.load(bf, allow_pickle=True)
            Xs.append(d["features"]); ys.append(d["labels"]); ps.append(d["paths"])
        X = np.concatenate(Xs).astype(np.float64)
        y = np.concatenate(ys).astype(np.int64)
        return X, y, np.concatenate(ps)


    X, y, ALL_PATHS = load_all_batches()
    n_real = int((y == 0).sum())
    n_ai   = int((y == 1).sum())
    logger.info("Feature matrix: X=%s  real=%d  AI=%d", X.shape, n_real, n_ai)

    assert X.shape[1] == N_FEATURES
    assert X.shape[0] == y.shape[0]
    if not np.all(np.isfinite(X)):
        bad = np.where(~np.isfinite(X).all(axis=1))[0]
        raise ValueError(f"Non-finite values in {len(bad)} rows — investigate.")
    logger.info("Feature matrix is finite and clean.")


    # ============================================================================
    # In[9]:  Class weight calculation
    # ============================================================================
    assert n_real > 0 and n_ai > 0, "Both classes must be present."
    weight_real = n_ai / n_real
    weight_ai   = 1.0
    CLASS_WEIGHT = {0: weight_real, 1: weight_ai}
    logger.info("Class weights: real=%.2f, AI=%.2f", weight_real, weight_ai)
    print(f"Class weights: real={weight_real:.2f}, AI={weight_ai:.2f}")


    # ============================================================================
    # In[10]:  LightGBM training with 5-fold CV
    # ============================================================================
    def fpr_at_tpr(y_true, scores, tpr_target=0.90):
        fpr, tpr, _ = roc_curve(y_true, scores)
        return float(fpr[min(np.searchsorted(tpr, tpr_target), len(fpr)-1)])

    def equal_error_rate(y_true, scores):
        fpr, tpr, _ = roc_curve(y_true, scores)
        fnr = 1.0 - tpr
        i = int(np.nanargmin(np.abs(fpr - fnr)))
        return float((fpr[i] + fnr[i]) / 2.0)

    def make_model():
        return LGBMClassifier(n_estimators=CONFIG.N_ESTIMATORS,
                              class_weight=CLASS_WEIGHT,
                              n_jobs=-1, random_state=CONFIG.SEED, verbose=-1)


    skf = StratifiedKFold(n_splits=CONFIG.N_SPLITS, shuffle=True,
                          random_state=CONFIG.SEED)
    fold_auc, fold_fpr90, fold_eer = [], [], []
    best_auc, best_bundle = -1.0, None

    logger.info("Starting %d-fold CV.", CONFIG.N_SPLITS)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        m = make_model(); m.fit(Xtr, y[tr])
        proba = m.predict_proba(Xva)[:, 1]
        auc  = roc_auc_score(y[va], proba)
        fpr  = fpr_at_tpr(y[va], proba)
        eer  = equal_error_rate(y[va], proba)
        fold_auc.append(auc); fold_fpr90.append(fpr); fold_eer.append(eer)
        logger.info("Fold %d: AUC=%.4f  FPR@90%%=%.4f  EER=%.4f", fold, auc, fpr, eer)
        if auc > best_auc:
            best_auc = auc
            best_bundle = {"model": m, "scaler": sc, "fold": fold,
                           "auc": auc, "feature_names": FEATURE_NAMES}

    cv_auc_mean, cv_auc_std   = float(np.mean(fold_auc)),  float(np.std(fold_auc))
    cv_fpr_mean, cv_fpr_std   = float(np.mean(fold_fpr90)),float(np.std(fold_fpr90))
    cv_eer_mean, cv_eer_std   = float(np.mean(fold_eer)),  float(np.std(fold_eer))
    logger.info("CV AUC=%.4f±%.4f  FPR@90%%=%.4f±%.4f  EER=%.4f±%.4f",
                cv_auc_mean, cv_auc_std, cv_fpr_mean, cv_fpr_std,
                cv_eer_mean, cv_eer_std)

    with open(CONFIG.BEST_FOLD_PATH, "wb") as f:
        pickle.dump(best_bundle, f)
    logger.info("Saved best fold model (fold %d, AUC=%.4f).", best_bundle["fold"],
                best_bundle["auc"])


    # ============================================================================
    # In[11]:  Train final model on the full dataset
    # ============================================================================
    logger.info("Training final model on all %d images.", X.shape[0])
    final_scaler = StandardScaler().fit(X)
    Xfull = final_scaler.transform(X)
    final_model = make_model(); final_model.fit(Xfull, y)

    with open(CONFIG.SCALER_PATH, "wb") as f:
        pickle.dump(final_scaler, f)
    with open(CONFIG.FINAL_MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "scaler": final_scaler,
                     "feature_names": FEATURE_NAMES,
                     "class_weight": CLASS_WEIGHT}, f)
    logger.info("Saved final model -> %s", CONFIG.FINAL_MODEL_PATH)


    # ============================================================================
    # In[12]:  Feature importance report
    # ============================================================================
    gain  = final_model.booster_.feature_importance(importance_type="gain")
    split = final_model.booster_.feature_importance(importance_type="split")
    order = np.argsort(gain)[::-1]
    top_features_by_gain = []
    logger.info("Feature importance (by gain):")
    for rank, i in enumerate(order, 1):
        logger.info("  %2d. %-22s gain=%12.1f  split=%6d",
                    rank, FEATURE_NAMES[i], gain[i], split[i])
        top_features_by_gain.append({"feature": FEATURE_NAMES[i],
                                     "gain": float(gain[i]),
                                     "split": int(split[i])})

    top_feature = FEATURE_NAMES[order[0]]
    bayer_confirmed = (top_feature == "ratio_var_cr")
    if not bayer_confirmed:
        logger.warning("Top feature is '%s', NOT 'ratio_var_cr'. "
                       "Expected from Bayer CFA theory — investigate.", top_feature)
    else:
        logger.info("Top feature 'ratio_var_cr' confirms Bayer CFA theory.")


    # ============================================================================
    # In[13]:  Save training report
    # ============================================================================
    hf_names = [s["name"] for s in CONFIG.HF_SOURCES]
    report = {
        "timestamp":          datetime.now().isoformat(),
        "n_real":             n_real,
        "n_ai":               n_ai,
        "n_total":            int(X.shape[0]),
        "real_sources":       [s["name"] for s in CONFIG.DATA_SOURCES if s["label"]==0],
        "ai_sources":         [s["name"] for s in CONFIG.DATA_SOURCES if s["label"]==1]
                              + hf_names,
        "cv_auc_mean":        cv_auc_mean,   "cv_auc_std":   cv_auc_std,
        "cv_fpr90_mean":      cv_fpr_mean,   "cv_fpr90_std": cv_fpr_std,
        "cv_eer_mean":        cv_eer_mean,   "cv_eer_std":   cv_eer_std,
        "top_features_by_gain": top_features_by_gain,
        "top_feature":        top_feature,
        "bayer_theory_confirmed": bayer_confirmed,
        "class_weights":      {"real": weight_real, "ai": weight_ai},
        "feature_names":      FEATURE_NAMES,
        "hyperparameters":    {"n_estimators": CONFIG.N_ESTIMATORS,
                               "n_splits": CONFIG.N_SPLITS,
                               "random_state": CONFIG.SEED},
        "model_path":         CONFIG.FINAL_MODEL_PATH.name,
    }
    with open(CONFIG.REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved training report -> %s", CONFIG.REPORT_PATH)


    # ============================================================================
    # In[14]:  Summary
    # ============================================================================
    bayer_msg = "confirms Bayer theory" if bayer_confirmed else "does NOT confirm Bayer theory"
    print("=" * 54)
    print("OPF TRAINING COMPLETE")
    print(f"Total images : {X.shape[0]:,}  ({n_real:,} real + {n_ai:,} AI)")
    print(f"CV AUC       : {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")
    print(f"CV FPR@90%   : {cv_fpr_mean:.4f} +/- {cv_fpr_std:.4f}")
    print(f"CV EER       : {cv_eer_mean:.4f} +/- {cv_eer_std:.4f}")
    print(f"Top feature  : {top_feature}  ({bayer_msg})")
    print(f"Model saved  : {CONFIG.FINAL_MODEL_PATH}")
    print("=" * 54)
    logger.info("Run finished cleanly.")


# ============================================================================
#  Entry point
# ============================================================================
def _setup_file_logging():
    CONFIG.WORKING_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    fh = logging.FileHandler(CONFIG.LOG_PATH, mode="a"); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "files", "stream", "train"],
                    default="all")
    args = ap.parse_args()
    _setup_file_logging()
    try:
        from huggingface_hub import get_token
        tok = get_token()
        if tok:
            os.environ.setdefault("HF_TOKEN", tok)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
        logger.info("HF token %s",
                    "present" if tok else "MISSING (streaming will be throttled)")
    except Exception as e:
        logger.warning("HF token check failed: %s", e)

    logger.info("=" * 70)
    logger.info("OPF LOCAL run — phase=%s", args.phase)
    logger.info("=" * 70)
    t0 = time.time()
    if args.phase in ("all", "files"):
        run_file_extraction(_load_manifest(), t0)
    if args.phase in ("all", "stream"):
        run_hf_streaming_parallel()
    if args.phase in ("all", "train"):
        run_training()
    logger.info("Phase '%s' finished in %.1f min", args.phase, (time.time() - t0) / 60)

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

