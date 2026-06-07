"""포폴/README용 히트맵 그리드 생성.

유형별로 'GT 마스크와 가장 잘 맞은' 결함 1장을 자동 선별해 그리드로 합친다.
선택 기준 = 이상 mass가 실제 결함 영역(GT 마스크) 안에 모인 비율(국소화 정밀도).
good은 가장 이상점수가 낮은(깨끗한) 1장.

렌더는 good/결함을 '공통 색 스케일'로 칠한다 → good은 차분, 결함만 뜨겁게(정직한 대조).

출력: results/figures/heatmap_portfolio.png (good+Type1+Type4+1, 1행)
      results/figures/heatmap_readme.png    (good+Type1~5, 2행)
실행: python build_heatmap_grid.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRUST_REMOTE_CODE", "1")
import sys, glob
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm

from mes import config
from mes.core.inference import ScrewInferencer

RES = config.IMAGE_SIZE
ROOT = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(ROOT, "data", "screw", "test")
GT = os.path.join(ROOT, "data", "screw", "ground_truth")
OUT_DIR = os.path.join(ROOT, "results", "figures")
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"

# 유형 폴더 → 라벨
LABELS = {
    "good":              "정상 (good)",
    "manipulated_front": "Type1 · 나사끝",
    "scratch_head":      "Type2 · 나사머리",
    "scratch_neck":      "Type3 · 나사목",
    "thread_side":       "Type4 · 나사산 측면",
    "thread_top":        "Type5 · 나사산 상단",
}
DEFECT_TYPES = ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB").resize((RES, RES)), dtype=np.float64)


def load_mask(folder, stem):
    mp = os.path.join(GT, folder, f"{stem}_mask.png")
    if not os.path.exists(mp):
        return None
    m = Image.open(mp).convert("L").resize((RES, RES))
    return np.asarray(m) > 127


def localization(amap, mask):
    """마스크 안/밖 평균 이상도의 비(농도비). 면적과 무관하게 '결함에 모였나'를 측정."""
    a = amap - amap.min()
    inside = a[mask].mean() if mask.any() else 0.0
    outside = a[~mask].mean() + 1e-9
    return float(inside / outside)


def render_cell(rgb, amap, vmin, vmax, gamma=2.5):
    """공통 스케일(vmin~vmax)로 jet 오버레이.

    alpha에 gamma 곡선을 줘 낮은 이상도(배경)는 원본을 거의 그대로 두고,
    높은 영역만 색이 뜨게 한다 → good은 차분, 결함만 부각(정직한 대조)."""
    a = np.clip((amap - vmin) / (vmax - vmin + 1e-9), 0, 1)
    heat = cm.jet(a)[..., :3] * 255.0
    alpha = (0.85 * (a ** gamma))[..., None]    # 배경 억제
    return ((1 - alpha) * rgb + alpha * heat).astype(np.uint8)


def pick():
    inf = ScrewInferencer()
    chosen = {}   # folder -> (path, stem, score, amap, rgb)

    # 결함: GT 일치도 최고 1장
    for folder in DEFECT_TYPES:
        best = None
        for p in sorted(glob.glob(os.path.join(TEST, folder, "*.png"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            mask = load_mask(folder, stem)
            if mask is None or not mask.any():
                continue
            b = inf.inferencer.predict(p)
            score = float(b.pred_score)
            if score < config.ANOMALY_THRESHOLD:      # 검출 못한 건 제외
                continue
            amap = b.anomaly_map.squeeze().detach().cpu().numpy()
            loc = localization(amap, mask)
            if best is None or loc > best[0]:
                best = (loc, p, stem, score, amap)
        if best:
            _, p, stem, score, amap = best
            chosen[folder] = (p, stem, score, amap, load_rgb(p))
            print(f"  {folder:18s} -> {os.path.basename(p)}  score={score:.3f}  국소화={best[0]:.2f}")

    # good: 이상점수 최저 1장
    best_good = None
    for p in sorted(glob.glob(os.path.join(TEST, "good", "*.png"))):
        b = inf.inferencer.predict(p)
        score = float(b.pred_score)
        if best_good is None or score < best_good[0]:
            amap = b.anomaly_map.squeeze().detach().cpu().numpy()
            best_good = (score, p, amap)
    score, p, amap = best_good
    chosen["good"] = (p, "good", score, amap, load_rgb(p))
    print(f"  {'good':18s} -> {os.path.basename(p)}  score={score:.3f}")
    return chosen


def compose(chosen, folders, out_path, ncol):
    """선택된 셀들을 공통 스케일로 렌더해 그리드 PNG로 저장."""
    # 공통 색 스케일: 결함 맵 기준 (good 포함 전체 min ~ 결함 99퍼센타일)
    defect_maps = [chosen[f][3] for f in folders if f != "good"]
    vmin = min(chosen[f][3].min() for f in folders)
    vmax = max(np.percentile(m, 99.5) for m in defect_maps)

    pad, label_h = 8, 34
    cell = RES
    nrow = (len(folders) + ncol - 1) // ncol
    W = ncol * cell + (ncol + 1) * pad
    H = nrow * (cell + label_h) + (nrow + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(FONT_PATH, 19)
    except Exception:
        font = ImageFont.load_default()

    for i, folder in enumerate(folders):
        r, c = divmod(i, ncol)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + label_h + pad)
        _, _, score, amap, rgb = chosen[folder]
        img = render_cell(rgb, amap, vmin, vmax)
        canvas.paste(Image.fromarray(img), (x, y))
        # 라벨 (유형 + 이상점수)
        txt = f"{LABELS[folder]}   ({score:.2f})"
        tb = draw.textbbox((0, 0), txt, font=font)
        tx = x + (cell - (tb[2] - tb[0])) // 2
        draw.text((tx, y + cell + 7), txt, fill=(30, 30, 30), font=font)

    os.makedirs(OUT_DIR, exist_ok=True)
    canvas.save(out_path)
    print(f"저장: {out_path}  ({W}x{H})")


def main():
    print("[선별] 유형별 best 샘플 (GT 국소화 기준)")
    chosen = pick()

    # README용: good + 5유형 (2행 x 3열)
    readme_order = ["good"] + DEFECT_TYPES
    compose(chosen, readme_order, os.path.join(OUT_DIR, "heatmap_readme.png"), ncol=3)

    # 포폴용: good + Type1 + Type4 + (Type2/3/5 중 국소화 최고 1개)
    extra = max(["scratch_head", "scratch_neck", "thread_top"],
                key=lambda f: localization(chosen[f][3], load_mask(f, chosen[f][1])))
    port_order = ["good", "manipulated_front", "thread_side", extra]
    compose(chosen, port_order, os.path.join(OUT_DIR, "heatmap_portfolio.png"), ncol=4)


if __name__ == "__main__":
    main()
