"""지도학습 baseline 의 데이터 접근 — `splits/baseline_split.csv` 매니페스트 기반.

baseline 은 MVTec screw 480장을 train 425 / test 55 로 고정 분할해 쓴다. 이미지는 `data/screw` 한 벌만 두고,
분할·유형(good/type1~5)·GT 마스크 대응은 매니페스트(split, class, filename, image, mask)로 표현한다.
(과거 `new_k-fold_data[_gt]` 폴더 복사본을 대체. filename 순 정렬 = 과거 sorted(os.listdir) 순서와 동일.)
"""
import os
import csv
import functools

from src.common.paths import BASELINE_SPLIT_CSV, MVTEC_DIR

CLASS_NAMES = ['good', 'type1', 'type2', 'type3', 'type4', 'type5']


@functools.lru_cache(maxsize=1)
def _load():
    """매니페스트 → (records, path→record 인덱스). records 는 (class 순서, filename) 정렬."""
    with open(BASELINE_SPLIT_CSV, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    order = {c: i for i, c in enumerate(CLASS_NAMES)}
    rows.sort(key=lambda r: (r['split'], order[r['class']], r['filename']))
    records = []
    for r in rows:
        records.append({
            'split': r['split'],
            'class': r['class'],
            'filename': r['filename'],
            'path': os.path.join(MVTEC_DIR, *r['image'].split('/')),
            'mask': os.path.join(MVTEC_DIR, *r['mask'].split('/')) if r['mask'] else None,
        })
    by_path = {os.path.normcase(r['path']): r for r in records}
    return records, by_path


def list_records(split, class_names=CLASS_NAMES):
    """split('train'|'test') 의 레코드를 class_names 순서 → filename 순으로 반환.
    각 레코드: split / class / filename / path(이미지 절대경로) / mask(GT 마스크 절대경로 또는 None)."""
    records, _ = _load()
    return [r for c in class_names for r in records if r['split'] == split and r['class'] == c]


def list_class(split, class_name):
    """한 클래스의 레코드(filename 순)."""
    return list_records(split, [class_name])


def class_of(path):
    """이미지 경로 → 클래스명('good'|'type1'..'type5'). 경로 문자열 파싱 대신 매니페스트로 판별한다."""
    return _load()[1][os.path.normcase(os.path.abspath(path))]['class']


def image_path(split, class_name, filename):
    """(split, class, filename) → 이미지 절대경로. filename 은 매니페스트의 filename 컬럼(예: '325.png')."""
    for r in list_class(split, class_name):
        if r['filename'] == filename:
            return r['path']
    raise FileNotFoundError(f'매니페스트에 없음: {split}/{class_name}/{filename}')
