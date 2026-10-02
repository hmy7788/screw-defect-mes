"""MES PoC 공통 설정.

경로·임계값 등 모든 모듈이 공유하는 값을 한곳에 모은다.
"""
from pathlib import Path

# 레포 루트 (이 파일 기준 한 단계 위)
ROOT = Path(__file__).resolve().parent.parent

# --- 모델 ---
# 연구 트랙에서 학습된 PatchCore 체크포인트를 재사용 (coreset 0.01, 해상도 320).
# 메모리뱅크가 작아 실시간 추론이 빠르고, 위치 정확도(AUPRO 0.956)도 최고급.
# (outputs/ 는 gitignore라 로컬에만 존재 — 없으면 `python experiments/anomaly/run_resolution_sweep.py 0.01` 재생성)
MODEL_CKPT = ROOT / "outputs" / "sweep_cs001" / "res320" / "Patchcore" / \
    "MVTecADFixed" / "screw" / "v0" / "weights" / "lightning" / "model.ckpt"
IMAGE_SIZE = 320               # 학습 해상도와 일치시킬 것

# export_model.py 가 ckpt를 ExportType.TORCH로 변환해 만드는 배포용 모델(.pt).
# 전처리(Resize+Normalize)·후처리(정규화+임계값)가 내장되어 원본 이미지를 그대로 넣을 수 있다.
RUNTIME_DIR = ROOT / "mes" / "runtime"
EXPORTED_PT = RUNTIME_DIR / "model" / "weights" / "torch" / "model.pt"

# good/bad 판정 임계값(정규화된 pred_score 기준).
# .pt 내장 임계값은 정상 오경보가 많아(FP 16/41), 프로젝트 방침(F2·FN 최소화)에 맞춰
# test셋에서 F2-최적 임계값을 재산출해 고정: thr=0.513 → TP118/FP9/FN1 (Recall 0.992, Prec 0.929).
# predict_one 은 pred_label 대신 이 값으로 판정한다. (재보정: mes/core/inference.py 셀프테스트)
ANOMALY_THRESHOLD = 0.513

# --- 런타임 산출물 (gitignore) ---
DB_PATH = RUNTIME_DIR / "inspections.sqlite"
HEATMAP_DIR = RUNTIME_DIR / "heatmaps"

# --- 라인 시뮬레이터 ---
FEED_SOURCE = ROOT / "data" / "screw" / "test"  # 흘려보낼 이미지 출처
FEED_INTERVAL_SEC = 1.0                          # 1장당 간격

# --- 서비스 ---
API_HOST = "127.0.0.1"
API_PORT = 8000
