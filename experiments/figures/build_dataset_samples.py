"""README 용 데이터셋 샘플 — 정상 + 결함 5유형(원본 위에 GT 마스크 윤곽선).

표본 규칙: baseline test split 의 유형별 첫 장(고정 규칙, build_comparison.py 와 동일).
출처: MVTec AD (CC BY-NC-SA 4.0), https://www.mvtec.com/company/research/datasets/mvtec-ad

실행: python experiments/figures/build_dataset_samples.py   → docs/images/dataset_samples.png
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.common.paths import REPO_ROOT
from src.supervised.data import list_records

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "Malgun Gothic", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
NAMES = {"good": "정상", "type1": "Type1\n나사끝", "type2": "Type2\n나사머리", "type3": "Type3\n나사목",
         "type4": "Type4\n나사산 측면", "type5": "Type5\n나사산 상단"}
RES = 360

fig, axes = plt.subplots(1, 6, figsize=(12.6, 2.9))
fig.subplots_adjust(top=0.74, bottom=0.03, left=0.01, right=0.99, wspace=0.04)
fig.text(0.012, 0.975, "데이터셋: MVTec AD screw — 정상 + 결함 5유형 (빨간 윤곽 = GT 마스크)", fontsize=13, fontweight="bold", color=INK, va="top")
for ax, cls in zip(axes, NAMES):
    rec = list_records("test", [cls])[0]
    img = Image.open(rec["path"]).convert("RGB").resize((RES, RES))
    ax.imshow(img)
    if rec["mask"]:
        m = (np.asarray(Image.open(rec["mask"]).convert("L").resize((RES, RES), Image.NEAREST)) > 127).astype(float)
        ax.contour(m, levels=[0.5], colors=["#e34948"], linewidths=1.8)
    ax.set_title(NAMES[cls], fontsize=10.5, color=INK, pad=5, linespacing=1.3)
    ax.axis("off")
out = os.path.join(REPO_ROOT, "docs", "images", "dataset_samples.png")
fig.savefig(out, dpi=140)
print("saved", os.path.relpath(out, REPO_ROOT))
