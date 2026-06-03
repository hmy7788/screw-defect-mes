"""MES PoC 공통 설정.

경로·임계값 등 모든 모듈이 공유하는 값을 한곳에 모은다.
"""
from pathlib import Path

# 레포 루트 (이 파일 기준 한 단계 위)
ROOT = Path(__file__).resolve().parent.parent

# --- 모델 ---
# 연구 트랙에서 학습된 PatchCore 체크포인트를 재사용.
# (outputs/ 는 gitignore라 로컬에만 존재 — 없으면 run_resolution_sweep.py 재실행으로 생성)
# TODO: 실제 ckpt 경로 확정 (cs01=coreset0.1 정밀 / cs001=0.01 빠름)
MODEL_CKPT = ROOT / "outputs" / "sweep_cs01" / "res320" / "Patchcore" / \
    "MVTecADFixed" / "screw" / "v0" / "weights" / "lightning" / "model.ckpt"
IMAGE_SIZE = 320               # 학습 해상도와 일치시킬 것
ANOMALY_THRESHOLD = None       # TODO: ad_eval 의 F2 튜닝 임계값을 고정값으로 박아넣기

# --- 런타임 산출물 (gitignore) ---
RUNTIME_DIR = ROOT / "mes" / "runtime"
DB_PATH = RUNTIME_DIR / "inspections.sqlite"
HEATMAP_DIR = RUNTIME_DIR / "heatmaps"

# --- 라인 시뮬레이터 ---
FEED_SOURCE = ROOT / "data" / "screw" / "test"  # 흘려보낼 이미지 출처
FEED_INTERVAL_SEC = 1.0                          # 1장당 간격

# --- 서비스 ---
API_HOST = "127.0.0.1"
API_PORT = 8000
