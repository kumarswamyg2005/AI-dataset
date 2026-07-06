# Are We Detecting AI or the Dataset?

Code and cached features for the paper
**"Are We Detecting AI or the Dataset? A Bias-Controlled Study of Frequency-Based AI-Image Detection"**
(submitted to IEEE WIFS 2026).

Every number in the paper regenerates from the scripts and result files in this repository.

## Reproduce Table I (no image download needed, CPU only, ~10 min)

```bash
pip install numpy opencv-python scikit-learn lightgbm
python corrected_artifacts/table1_final.py
```

DCT features for all six GenImage families are cached in `corrected_artifacts/cache_*/`
(`.npz` files), so this reproduces every "Ours" number in Table I from scratch —
including the bootstrap confidence intervals. Expected output is saved in
`corrected_artifacts/table1_final_results.txt`.

> Note: the scripts contain absolute paths from the authors' machine at the top;
> point them at your own copy of this repository before running.

## Reproduce from raw images (optional)

1. Download GenImage: https://github.com/GenImage-Dataset/GenImage
   (we use the ImageNet-class subsets: SD v1.5, ADM, GLIDE, Midjourney, Wukong, VQDM).
2. Edit the data paths at the top of the scripts.
3. Re-run `table1_final.py` — features are re-extracted under the paper's protocol
   (native 256x256 center crop, no resize, re-encode JPEG Q95 4:4:4).

We do not redistribute the images themselves (GenImage / ImageNet licenses).

## Map: paper number -> script -> result file

| Paper | Script | Result file |
|---|---|---|
| Table I, "Ours" columns + CIs | `table1_final.py` | `table1_final_results.txt` |
| Table I, zero-shot NPR / FatFormer + NPR-ft | `deep_full_chain.py` | `deep_full_results.txt`, `npr_finetuned_results.txt` |
| Finding 2: Base-189, score-avg fusion | `jpeg_control_and_base189.py` | `jpeg_control_and_base189_results.txt` |
| Finding 4: history probe (variant 1) + matched-history control | `jpeg_control_and_base189.py` | `jpeg_control_and_base189_results.txt` |
| Finding 4: history variant 2 (libjpeg Q70-90) | `history_variant2.py` | `history_variant2_results.txt` |
| SD-Turbo row | `sdxl_row.py` (generate+features), `eval_sd_turbo_row.py` (eval) | `sd_turbo_row_results.txt` |
| Figures 1 and 2 | `figs_corrected.py` (repo root) | `figs/` |

`opf_local_train.py` (repo root) is the OPF feature extractor used by all scripts.
`deep_scores*/` hold the raw per-image scores of NPR and FatFormer.
`kaggle_sdxl_generate.py` is a recipe to regenerate the SD-Turbo-style fakes on a free GPU;
`sdxl_row.py` uses fixed seeds, so the fakes are reproducible.

## The protocol in one paragraph

Real images are ImageNet photographs; fakes are GenImage generations of the same object
classes (content-matched). Every image gets a native 256x256 center crop (no resizing) and
is re-encoded at JPEG quality 95 with 4:4:4 chroma, identically for both classes. Training
is leave-one-generator-out; the test generator is never seen. AUC with the AI class positive;
95% bootstrap CIs (2000 resamples). Classifier everywhere: LightGBM, 1000 trees, 63 leaves,
learning rate 0.03, balanced classes, seed 42.

## License

MIT — see `LICENSE`.
