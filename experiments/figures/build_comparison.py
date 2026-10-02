"""baseline(ResNet-18 + Grad-CAM) vs PatchCore — 같은 이미지에서 결함 위치를 나란히 비교하는 그림.

  docs/images/compare_baseline_vs_patchcore.png   good + Type1~5 (6행)
  docs/images/compare_type4.png                   Type4 test 3장 전부 (baseline 이 가장 약한 유형)

표본 선정에 편향이 없도록 **고정 규칙**을 쓴다: baseline 의 test split(splits/baseline_split.csv)에서
각 유형의 첫 장(filename 순). PatchCore 가 잘 맞는 장면만 고르지 않는다.
baseline 은 학습에 쓰지 않은 test 이미지, PatchCore 는 정상(train/good)만 학습한 모델이라 둘 다 공정하게 unseen.

열: 원본 | GT 마스크 | baseline Grad-CAM(class=bad) | PatchCore anomaly map
  - 두 히트맵은 같은 방식(min-max 정규화 후 jet, 이상도 비례 alpha)으로 칠한다 → 색 비교가 정직하다.
  - baseline 은 입력을 Resize(256)+CenterCrop(224)로 보므로 중앙 영역 밖은 정보가 없다(가장자리는 최저값으로 채움).

선행: outputs/supervised/best_model.pth (baseline), PatchCore 배포 모델 .pt (python -m mes.core.export_model)
실행: python experiments/figures/build_comparison.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRUST_REMOTE_CODE", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from mes import config
from src.common.paths import REPO_ROOT, SUPERVISED_OUT_DIR
from src.supervised.data import list_records
from src.supervised.dataset import val_transform
from src.supervised.model import build_model_bin

OUT_DIR = os.path.join(REPO_ROOT, "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

RES = 320                      # 표시 해상도 (PatchCore 입력 해상도와 동일)
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
TYPE_NAMES = {"good": "정상 (good)", "type1": "Type1 · 나사끝", "type2": "Type2 · 나사머리",
              "type3": "Type3 · 나사목", "type4": "Type4 · 나사산 측면", "type5": "Type5 · 나사산 상단"}

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def overlay(rgb, amap):
    """mes.core.inference.render_heatmap 과 같은 합성(min-max → jet, 이상도 비례 alpha)을 배열로 반환."""
    a = np.asarray(amap, dtype=np.float64)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    heat = cm.jet(a)[..., :3] * 255.0
    alpha = (0.75 * a)[..., None]
    return ((1 - alpha) * np.asarray(rgb, dtype=np.float64) + alpha * heat).astype(np.uint8)


def load_baseline(device):
    model = build_model_bin("resnet18")
    model.load_state_dict(torch.load(os.path.join(SUPERVISED_OUT_DIR, "best_model.pth"), map_location=device))
    return model.to(device).eval()


def baseline_cam(model, cam, path, device):
    """Grad-CAM(class=bad, 224×224 중앙 crop 기준) → 320×320 전체 화면에 올린 맵 + bad 확률."""
    img = Image.open(path).convert("RGB")
    x = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p_bad = torch.softmax(model(x), dim=1)[0, 1].item()
    g = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]            # (224,224), no_grad 밖
    full = np.zeros((256, 256), dtype=np.float32)
    full[16:240, 16:240] = g                                                    # CenterCrop(224) 위치
    full = np.asarray(Image.fromarray(full).resize((RES, RES), Image.BILINEAR))
    return full, p_bad


def load_mask(mask_path):
    if not mask_path:
        return np.zeros((RES, RES), dtype=np.uint8)
    m = Image.open(mask_path).convert("L").resize((RES, RES), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def make_grid(records, out_name, title, subtitle):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_baseline(device)
    cam = GradCAM(model=base, target_layers=[base.layer4[-1]])
    from anomalib.deploy import TorchInferencer
    pc = TorchInferencer(str(config.EXPORTED_PT), device="auto")

    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(9.6, 2.45 * n + 1.1))
    fig.subplots_adjust(top=1 - 0.95 / (2.45 * n + 1.1), bottom=0.01, left=0.075, right=0.995, wspace=0.03, hspace=0.07)
    top = 1 - 0.08 / (2.45 * n + 1.1) * 3
    fig.text(0.03, 0.992, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.992 - 0.34 / (2.45 * n + 1.1), subtitle, fontsize=9.8, color=INK2, va="top")
    if n == 1:
        axes = axes[None, :]
    col_titles = ["원본", "GT 마스크 (실제 결함 위치)", "baseline: ResNet-18 + Grad-CAM", "PatchCore: 이상 점수 맵"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10.5, fontweight="bold", color=INK, pad=6)

    for r, rec in enumerate(records):
        path = rec["path"]
        rgb = np.asarray(Image.open(path).convert("RGB").resize((RES, RES)))
        mask = load_mask(rec["mask"])
        g, p_bad = baseline_cam(base, cam, path, device)
        batch = pc.predict(path)
        amap = batch.anomaly_map.squeeze().detach().cpu().numpy()
        score = float(batch.pred_score)

        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(overlay(rgb, g))
        axes[r, 3].imshow(overlay(rgb, amap))
        verdict = "bad" if score >= config.ANOMALY_THRESHOLD else "good"
        axes[r, 2].text(0.5, -0.02, f"bad 확률 {p_bad:.2f}", transform=axes[r, 2].transAxes, ha="center", va="top",
                        fontsize=8.8, color=INK2)
        axes[r, 3].text(0.5, -0.02, f"이상 점수 {score:.2f} → {verdict}", transform=axes[r, 3].transAxes, ha="center",
                        va="top", fontsize=8.8, color=INK2)
        axes[r, 0].text(-0.06, 0.5, rec["label"], transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center",
                        fontsize=10.5, fontweight="bold", color=INK)
        for c in range(4):
            axes[r, c].axis("off")
    path_out = os.path.join(OUT_DIR, out_name)
    fig.savefig(path_out, dpi=130)
    plt.close(fig)
    print("saved", os.path.relpath(path_out, REPO_ROOT))


def first_of(cls):
    return list_records("test", [cls])[0]


if __name__ == "__main__":
    # 1) good + Type1~5, 각 유형 test 첫 장
    rows = []
    for cls in ["good", "type1", "type2", "type3", "type4", "type5"]:
        r = dict(first_of(cls)); r["label"] = TYPE_NAMES[cls] + f"\n{r['filename']}"
        rows.append(r)
    make_grid(rows, "compare_baseline_vs_patchcore.png",
              "같은 이미지, 다른 결과 — 결함 위치를 누가 짚는가",
              "baseline test 의 유형별 첫 장(선별 없음). 빨간색일수록 \"여기가 결함\"이라는 모델의 근거.")
    # 2) Type4 전부
    t4 = []
    for r in list_records("test", ["type4"]):
        r = dict(r); r["label"] = f"Type4\n{r['filename']}"
        t4.append(r)
    make_grid(t4, "compare_type4.png",
              "가장 어려운 Type4(나사산 측면) — baseline test 3장 전부",
              "baseline 은 Type4 Pixel AUROC 0.65 / AUPRO 0.45. 같은 3장에서 PatchCore 가 어디를 짚는지 비교.")
