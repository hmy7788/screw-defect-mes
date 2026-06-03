"""이상탐지(anomaly detection) 트랙 — 모델 빌드 + 학습/평가 래퍼.

anomalib Engine 기반. PatchCore를 기본으로, EfficientAD/PaDiM도 동일 인터페이스로 빌드.
PatchCore는 epoch 학습이 아니라 good 특징으로 메모리뱅크를 구성한다.
모델 기본 pre_processor가 입력을 256x256으로 리사이즈 + ImageNet 정규화한다.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # OMP libiomp5md 중복 회피

from anomalib.engine import Engine


def build_model(name="patchcore", **kwargs):
    """이상탐지 모델 생성. name: patchcore | efficient_ad | padim."""
    name = name.lower()
    if name == "patchcore":
        from anomalib.models import Patchcore
        defaults = dict(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )
        defaults.update(kwargs)
        return Patchcore(**defaults)
    if name in ("efficient_ad", "efficientad"):
        from anomalib.models import EfficientAd
        return EfficientAd(**kwargs)
    if name == "padim":
        from anomalib.models import Padim
        return Padim(**kwargs)
    raise ValueError(f"Unknown model: {name}")


def run_experiment(model, datamodule, output_dir, accelerator="gpu", devices=1, max_epochs=1):
    """학습(fit) 후 test 평가. (engine, test_results) 반환.

    test_results: anomalib가 계산한 지표 dict 리스트(image/pixel AUROC, F1 등).
    결함 위치 히트맵은 output_dir 하위에 자동 저장(visualizer=True).
    """
    os.makedirs(output_dir, exist_ok=True)
    engine = Engine(
        default_root_dir=output_dir,
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
    )
    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)
    return engine, results
