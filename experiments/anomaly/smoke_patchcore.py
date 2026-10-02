"""PatchCore 스모크런 — 파이프라인 전 구간(학습→평가→위치 히트맵)이 도는지 1회 확인.

설정: PatchCore(WideResNet-50, layer2+3), coreset 10%, 입력 256.
data/screw (train good 320 / test good 41 + 결함 119, 마스크 포함) 사용.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # 레포 루트
from src.common.paths import REPO_ROOT, DATA_DIR, OUTPUTS_DIR, RESULTS_DIR
from src.anomaly.ad_data import build_screw_datamodule
from src.anomaly.models import build_model, run_experiment

torch.manual_seed(42)

OUT = os.path.join(OUTPUTS_DIR, "smoke_patchcore")

print("=" * 60)
print("PatchCore 스모크런 시작")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("=" * 60)

dm = build_screw_datamodule(root=DATA_DIR, category="screw",
                            train_batch_size=8, eval_batch_size=8, num_workers=0, seed=42)
dm.setup()
print(f"데이터 - train:{len(dm.train_data)}  val:{len(dm.val_data)}  test:{len(dm.test_data)}")

model = build_model("patchcore", coreset_sampling_ratio=0.1)

t0 = time.time()
engine, results = run_experiment(model, dm, output_dir=OUT, accelerator="gpu", devices=1)
dt = time.time() - t0

print("\n" + "=" * 60)
print(f"완료 - 총 {dt:.1f}초")
print("test 지표:")
for r in results:
    for k, v in r.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
print(f"\n결과/히트맵 저장 위치: {OUT}")
print("=" * 60)
