"""README 용 차트 4종 생성 — results/ 의 CSV 만 읽는다(새 실험 없음).

  chart_model_compare.png         모델 비교 (이미지 AUROC / 위치 AUPRO @320)
  chart_resolution_type4.png      해상도 256→320 효과 (유형별 위치, PatchCore coreset 0.1)
  chart_coreset_tradeoff.png      coreset 0.1 vs 0.01 — 정확도(AUPRO) vs 추론 지연
  chart_baseline_vs_patchcore.png baseline(Grad-CAM) vs PatchCore — 유형별 위치(Pixel AUROC)

색 규칙(dataviz 스킬): 엔티티 고정색 — PatchCore=blue, EfficientAD=orange, PaDiM=aqua, baseline=중립 회색.
해상도처럼 순서가 있는 값은 blue 단일 hue 의 밝음→어두움. 텍스트는 series 색이 아니라 text 토큰 색.

실행: python experiments/figures/build_charts.py   → docs/images/
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common.paths import RESULTS_DIR, REPO_ROOT

OUT_DIR = os.path.join(REPO_ROOT, "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)
R = lambda *p: os.path.join(RESULTS_DIR, *p)

# --- 토큰 (palette.md 의 light 값) ---
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e7e6e2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#1c5cab"      # 순서형(해상도) 램프 step 250 / 550
BASELINE_GRAY = "#8a8984"

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1,
})


def _csv(path):
    df = pd.read_csv(path, skipinitialspace=True, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    return df


def _clean_axes(ax, grid_axis="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _title(fig, title, subtitle):
    fig.text(0.04, 0.965, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.905, subtitle, fontsize=10.5, color=INK2, va="top")


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("saved", os.path.relpath(path, REPO_ROOT))


# ------------------------------------------------------------------ A. 모델 비교
def chart_model_compare():
    s = _csv(R("anomaly", "model_compare", "summary.csv"))
    s = s[s["res"] == 320].set_index("model")
    models = [("patchcore", "PatchCore", BLUE), ("efficient_ad", "EfficientAD", ORANGE), ("padim", "PaDiM", AQUA)]
    panels = [("image_AUROC", "이미지 단위 검출 (Image AUROC)"), ("AUPRO", "결함 위치 (AUPRO)")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.subplots_adjust(top=0.70, bottom=0.14, left=0.06, right=0.97, wspace=0.28)
    _title(fig, "모델 비교 — PatchCore 가 검출·위치 양면 최선",
           "MVTec screw, 해상도 320, 정상만 학습. 막대 위 숫자 = 지표값 (1.0 에 가까울수록 좋음)")
    for ax, (col, ttl) in zip(axes, panels):
        for i, (key, label, color) in enumerate(models):
            v = float(s.loc[key, col])
            ax.bar(i, v, width=0.5, color=color, zorder=3)
            ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels([m[1] for m in models], fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(ttl, fontsize=11.5, color=INK, loc="left", pad=10)
        _clean_axes(ax)
    fig.text(0.04, 0.02, "EfficientAD 는 20 epoch 간이 학습 조건. PaDiM 은 불량을 전부 잡지만(Recall 1.0) 정상 오경보가 많아 AUROC 가 낮다.",
             fontsize=9, color=MUTED)
    _save(fig, "chart_model_compare.png")


# ------------------------------------------------------------------ B. 해상도 효과
def chart_resolution_type4():
    a = _csv(R("anomaly", "sweep_cs01", "per_type_res256.csv"))
    b = _csv(R("anomaly", "sweep_cs01", "per_type_res320.csv"))
    names = {"Type1": "Type1 · 나사끝", "Type2": "Type2 · 나사머리", "Type3": "Type3 · 나사목",
             "Type4": "Type4 · 나사산 측면", "Type5": "Type5 · 나사산 상단"}
    key = [c for c in a.columns if "pixel" in c.lower()][0]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.19, right=0.95)
    _title(fig, "해상도를 높이면 Type4(나사산 측면) 위치가 가장 크게 좋아진다",
           "PatchCore(coreset 0.1) 유형별 위치 정확도 (Pixel AUROC), 256 → 320")
    n = len(a)
    for i, t in enumerate(a["Type"]):
        y = n - 1 - i
        v1, v2 = float(a.loc[a["Type"] == t, key].iloc[0]), float(b.loc[b["Type"] == t, key].iloc[0])
        ax.plot([v1, v2], [y, y], color=GRID if t != "Type4" else INK2, linewidth=2 if t == "Type4" else 1.5, zorder=2, solid_capstyle="round")
        ax.scatter([v1], [y], s=110, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([v2], [y], s=110, color=BLUE_DARK, edgecolor=SURFACE, linewidth=2, zorder=3)
        if t == "Type4":
            ax.text(v1, y + 0.26, f"{v1:.3f}", ha="center", va="bottom", fontsize=10.5, color=INK2)
            ax.text(v2, y + 0.26, f"{v2:.3f}", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
            ax.text((v1 + v2) / 2, y - 0.30, f"+{v2 - v1:.3f} (최대 개선)", ha="center", va="top", fontsize=10.5,
                    color=INK, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels([names[t] for t in a["Type"]][::-1], fontsize=11)
    ax.set_xlim(0.965, 1.0)
    ax.set_ylim(-0.8, n - 0.3)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    _clean_axes(ax, "x")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=BLUE_LIGHT, label="해상도 256"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=BLUE_DARK, label="해상도 320")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10, labelcolor=INK2)
    _save(fig, "chart_resolution_type4.png")


# ------------------------------------------------------------------ C. coreset 트레이드오프
def chart_coreset_tradeoff():
    d = _csv(R("anomaly", "patchcore_all.csv"))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.09, right=0.96)
    _title(fig, "coreset 0.01 은 위치 정확도를 거의 지키면서 2배 빠르다",
           "PatchCore 4개 변형 — 가로: 추론 지연(p50, 낮을수록 좋음), 세로: 결함 위치 정확도(AUPRO, 높을수록 좋음)")
    offs = {"cs01_256": (10, -14), "cs01_320": (-10, -20), "cs001_256": (12, 6), "cs001_320": (14, 8)}
    for _, r in d.iterrows():
        v = r["variant"]
        adopted = v == "cs001_320"
        ax.scatter([r["p50_ms"]], [r["AUPRO"]], s=300 if adopted else 130, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        label = f"coreset {r['coreset']} · {int(r['res'])}px\n{r['p50_ms']:.0f}ms ({r['fps']:.0f} FPS) · AUPRO {r['AUPRO']:.3f}"
        if adopted:
            label = "MES 채택\n" + label
        dx, dy = offs[v]
        ax.annotate(label, (r["p50_ms"], r["AUPRO"]), textcoords="offset points", xytext=(dx, dy),
                    ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top", fontsize=9.5,
                    color=INK, fontweight="bold" if adopted else "normal")
    ax.set_xlim(25, 95)
    ax.set_ylim(0.895, 0.98)
    ax.set_xlabel("추론 지연 p50 (ms)", fontsize=10)
    ax.set_ylabel("AUPRO", fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=1)
    _clean_axes(ax, "y")
    ax.grid(axis="x", color=GRID, linewidth=1)
    _save(fig, "chart_coreset_tradeoff.png")


# ------------------------------------------------------------------ D. baseline vs PatchCore
def chart_baseline_vs_patchcore():
    base = _csv(R("supervised", "localization.csv"))
    base = base[base["type"].str.startswith("type")].copy()
    base["Type"] = base["type"].str.replace("type", "Type")
    pc = _csv(R("anomaly", "sweep_cs001", "per_type_res320.csv"))
    key = [c for c in pc.columns if "pixel" in c.lower()][0]
    names = {"Type1": "Type1 · 나사끝", "Type2": "Type2 · 나사머리", "Type3": "Type3 · 나사목",
             "Type4": "Type4 · 나사산 측면", "Type5": "Type5 · 나사산 상단"}
    fig, ax = plt.subplots(figsize=(10, 4.9))
    fig.subplots_adjust(top=0.80, bottom=0.2, left=0.19, right=0.95)
    _title(fig, "분류기(Grad-CAM)는 Type4 위치를 못 짚고, 이상탐지는 짚는다",
           "유형별 결함 위치 정확도 (Pixel AUROC, 1.0 에 가까울수록 좋음)")
    n = 5
    for i, t in enumerate(names):
        y = n - 1 - i
        vb = float(base.loc[base["Type"] == t, "pixel_auroc"].iloc[0])
        vp = float(pc.loc[pc["Type"] == t, key].iloc[0])
        hot = t == "Type4"
        ax.plot([vb, vp], [y, y], color=INK2 if hot else GRID, linewidth=2 if hot else 1.5, zorder=2, solid_capstyle="round")
        ax.scatter([vb], [y], s=120, color=BASELINE_GRAY, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([vp], [y], s=120, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        if hot:
            ax.text(vb - 0.012, y, f"{vb:.2f}", ha="right", va="center", fontsize=11, color=INK, fontweight="bold")
            ax.text(vp + 0.012, y, f"{vp:.2f}", ha="left", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels([names[t] for t in names][::-1], fontsize=11)
    ax.set_xlim(0.6, 1.03)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    _clean_axes(ax, "x")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=BASELINE_GRAY, label="baseline: ResNet-18 + Grad-CAM"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=BLUE, label="PatchCore (coreset 0.01, 320)")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=10, labelcolor=INK2)
    fig.text(0.04, 0.03, "주의: 두 실험은 split 이 달라 엄밀한 직접 비교는 아니다 — baseline 은 test 결함 19장(Type4 는 3장), PatchCore 는 결함 119장(Type4 는 23장).",
             fontsize=9, color=MUTED)
    _save(fig, "chart_baseline_vs_patchcore.png")


if __name__ == "__main__":
    chart_model_compare()
    chart_resolution_type4()
    chart_coreset_tradeoff()
    chart_baseline_vs_patchcore()
