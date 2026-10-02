"""이상탐지(anomaly detection) 트랙용 데이터 모듈.

MVTec AD `screw` 카테고리(원본 + ground_truth 마스크)를 anomalib으로 로드한다.

주의 — anomalib 2.5.0 + Python 3.11 호환 버그 우회:
    anomalib의 `Split` enum은 `(str, Enum)` 기반인데, Python 3.11에서 이 enum을
    pandas 컬럼 필터(`samples.split == Split.TRAIN`)에 쓰면 0건이 매칭되어
    데이터셋이 빈 채로 로드된다(스칼라 비교 `Split.TRAIN == 'train'`은 True지만
    pandas 벡터 비교는 실패). 아래 `MVTecADFixed`는 `_setup`에서 split을
    문자열('train'/'test')로 넘겨 이 문제를 우회한다. 검증: train 320 / test 160 정상.
"""

import os

from anomalib.data import MVTecAD
from src.common.paths import DATA_DIR
from anomalib.data.datasets.image.mvtecad import MVTecADDataset


class MVTecADFixed(MVTecAD):
    """split을 문자열로 전달해 Python 3.11 enum 필터 버그를 우회한 MVTec 데이터 모듈."""

    def _setup(self, _stage=None):
        self.train_data = MVTecADDataset(
            split="train", root=self.root, category=self.category
        )
        self.test_data = MVTecADDataset(
            split="test", root=self.root, category=self.category
        )


def _anomalib_root(path):
    """anomalib 은 데이터 경로에 ASCII 출력가능 문자([ -~])만 허용한다(`validate_path`).
    한글 사용자 폴더 등 비ASCII 절대경로는 거부되므로, cwd 기준 상대경로로 넘긴다(레포 루트에서 실행하면 'data').
    다른 드라이브라 상대경로를 만들 수 없으면 절대경로를 그대로 쓴다."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def build_screw_datamodule(
    root=DATA_DIR,
    category="screw",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,          # Windows/Jupyter는 0 권장
    seed=42,
    **kwargs,
):
    """screw 이상탐지용 데이터 모듈 생성.

    기본 분할(anomalib 표준 MVTec 프로토콜): train=good 320,
    test=good 41 + 결함 119(마스크 포함), val=test와 동일(SAME_AS_TEST).
    임계값을 train good에서 따로 잡고 싶으면 val_split_mode 등을 kwargs로 조정.
    """
    return MVTecADFixed(
        root=_anomalib_root(root),
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        seed=seed,
        **kwargs,
    )
