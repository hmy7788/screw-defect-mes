"""레포 공용 경로 — 모든 트랙(supervised / anomaly)이 같은 기준을 쓴다 (실행 위치와 무관).

  data/screw/         MVTec AD screw 원본 (gitignore)  ← 데이터는 이 한 벌만 둔다
  splits/             분할 매니페스트 (추적)            ← baseline 의 train/test 분할은 여기서 정의
  outputs/            학습 산출물 (gitignore)
  results/            커밋하는 결과 (supervised/, anomaly/)
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_DIR = os.path.join(REPO_ROOT, 'data')
MVTEC_DIR = os.path.join(DATA_DIR, 'screw')

SPLITS_DIR = os.path.join(REPO_ROOT, 'splits')
BASELINE_SPLIT_CSV = os.path.join(SPLITS_DIR, 'baseline_split.csv')

OUTPUTS_DIR = os.path.join(REPO_ROOT, 'outputs')
SUPERVISED_OUT_DIR = os.path.join(OUTPUTS_DIR, 'supervised')

RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
