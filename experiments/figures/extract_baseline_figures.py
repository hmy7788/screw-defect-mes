"""baseline(졸업논문) 그림을 README 용으로 정리 — 논문 PDF 의 그림 추출 + 위치평가 그림 축소.

  docs/images/baseline_augmentation.png      논문 그림: 학습 증강 예시 (8쪽)
  docs/images/baseline_loss_curves.png       논문 그림: ResNet-18 / MobileNet-V2 / VGG-16 학습·검증 손실 (13쪽, 3장 병합)
  docs/images/baseline_gradcam_resnet18.png  논문 그림: ResNet-18 예측 vs Grad-CAM (16쪽)
  docs/images/baseline_localization_gt.png   Grad-CAM vs GT 마스크 (localization_eval.py 산출물 축소; 선행: 위치평가 실행)

실행: python experiments/figures/extract_baseline_figures.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pymupdf
from PIL import Image

from src.common.paths import REPO_ROOT, SUPERVISED_OUT_DIR

OUT = os.path.join(REPO_ROOT, "docs", "images")
os.makedirs(OUT, exist_ok=True)
PDF = glob.glob(os.path.join(REPO_ROOT, "docs", "thesis", "*.pdf"))[0]


def page_images(doc, page):
    imgs = []
    for info in doc[page - 1].get_images(full=True):
        pix = pymupdf.Pixmap(doc, info[0])
        if pix.n > 4:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return imgs


def save(im, name, width=None):
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    path = os.path.join(OUT, name)
    im.save(path, optimize=True)
    print("saved", os.path.relpath(path, REPO_ROOT), im.size, os.path.getsize(path) // 1024, "KB")


doc = pymupdf.open(PDF)
save(page_images(doc, 8)[0], "baseline_augmentation.png")
curves = page_images(doc, 13)                       # ResNet-18, MobileNet-V2, VGG-16 순
strip = Image.new("RGB", (sum(i.width for i in curves), max(i.height for i in curves)), "white")
x = 0
for i in curves:
    strip.paste(i, (x, 0)); x += i.width
save(strip, "baseline_loss_curves.png")
save(page_images(doc, 16)[0], "baseline_gradcam_resnet18.png")
save(Image.open(os.path.join(SUPERVISED_OUT_DIR, "ResNet-18_localization_grid.png")).convert("RGB"),
     "baseline_localization_gt.png", 830)
