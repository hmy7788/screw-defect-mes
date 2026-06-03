"""학습된 PatchCore 체크포인트를 배포용 .pt 로 export (1회성).

anomalib Engine.export(ExportType.TORCH)는 전체 LightningModule을 저장하며,
모듈 forward가 pre_processor(Resize+Normalize) → model → post_processor(정규화+임계값)를
모두 적용한다. 따라서 결과 .pt 는 원본 이미지를 그대로 넣어도 동작한다(TorchInferencer용).

실행: python -m mes.core.export_model
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.deploy import ExportType
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, Normalize

from mes import config

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_pp(res):
    """학습(run_resolution_sweep.py)과 동일한 전처리: Resize + ImageNet 정규화."""
    return PreProcessor(transform=Compose([
        Resize((res, res), antialias=True),
        Normalize(mean=MEAN, std=STD),
    ]))


def main():
    if not config.MODEL_CKPT.exists():
        print(f"[오류] ckpt 없음: {config.MODEL_CKPT}")
        print("  -> `python run_resolution_sweep.py 0.01` 로 먼저 학습/생성하세요.")
        sys.exit(1)

    # 학습과 동일 설정으로 모델 골격 빌드 (가중치/메모리뱅크는 ckpt_path에서 로드됨)
    model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                      coreset_sampling_ratio=0.01, pre_processor=make_pp(config.IMAGE_SIZE))
    engine = Engine(accelerator="gpu", devices=1)

    export_root = config.RUNTIME_DIR / "model"
    print(f"[export] ckpt={config.MODEL_CKPT}")
    print(f"[export] -> {export_root}")
    out = engine.export(model=model, export_type=ExportType.TORCH,
                        ckpt_path=str(config.MODEL_CKPT), export_root=str(export_root))
    print(f"[완료] 생성된 .pt: {out}")
    print(f"[확인] config.EXPORTED_PT = {config.EXPORTED_PT}  존재={config.EXPORTED_PT.exists()}")


if __name__ == "__main__":
    main()
