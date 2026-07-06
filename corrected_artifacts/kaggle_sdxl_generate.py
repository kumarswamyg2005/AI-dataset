"""
KAGGLE RECIPE — generate the SDXL-Turbo fakes for the modern-generator row on a free GPU.
(The local Mac has 8 GB RAM; SDXL-Turbo needs a ~16 GB GPU, so this runs on Kaggle instead.)

How to use (caveman steps):
 1. On the Mac:  cd /Users/kumaraswamy/Desktop/jpg/research/data/norm256/imagenet_midjourney
                 zip -r mj_nature.zip nature   (only ~35 MB)
    Upload mj_nature.zip as a private Kaggle dataset called "mj-nature-norm256".
 2. New Kaggle notebook, GPU T4 x1, attach that dataset, paste this whole file, Run All.
    Takes roughly 30-45 minutes for 1000 images.
 3. Download the output sdxl_turbo_ai.zip from the notebook's /kaggle/working.
 4. On the Mac: unzip into /Users/kumaraswamy/Desktop/jpg/research/data/sdxl_turbo/ai/
    then run:  GEN_MODEL="stabilityai/sdxl-turbo" GEN_TAG="sdxl_turbo" \
               .venv/bin/python corrected_artifacts/sdxl_row.py
    (generation is skipped because the PNGs already exist; it only extracts features
     and evaluates, ~10 minutes on CPU) -> sdxl_turbo_row_results.txt
"""
import os, glob, json, zipfile, urllib.request
import torch
from diffusers import AutoPipelineForText2Image

N = 1000
SRC = "/kaggle/input/mj-nature-norm256/nature"      # the uploaded real images (for class list)
OUT = "/kaggle/working/sdxl_turbo_ai"; os.makedirs(OUT, exist_ok=True)

url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
w2n = {v[0]: v[1].replace("_", " ") for v in json.load(urllib.request.urlopen(url)).values()}
reals = sorted(glob.glob(f"{SRC}/*.jpg"))[:N]
classes = [w2n[os.path.basename(p).split("_")[0]] for p in reals]
print(f"{len(classes)} prompts from {len(reals)} real images")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.set_progress_bar_config(disable=True)

for i, c in enumerate(classes):
    fp = f"{OUT}/{i:04d}.png"
    if os.path.exists(fp): continue
    g = torch.Generator("cpu").manual_seed(i)   # same seeds as the local script
    img = pipe(prompt=f"photo of {c}", num_inference_steps=2, guidance_scale=0.0,
               height=512, width=512, generator=g).images[0]
    img.save(fp)
    if (i + 1) % 100 == 0: print(f"{i+1}/{len(classes)}")

with zipfile.ZipFile("/kaggle/working/sdxl_turbo_ai.zip", "w") as z:
    for p in sorted(glob.glob(f"{OUT}/*.png")):
        z.write(p, f"ai/{os.path.basename(p)}")
print("DONE -> download /kaggle/working/sdxl_turbo_ai.zip")
