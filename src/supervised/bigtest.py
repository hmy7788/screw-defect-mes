"""test 결함 수를 늘린 분할 — baseline 480장(train 425 + test 55)을 합쳐 test 를 다시 뽑는다.

왜: baseline test 결함은 19장뿐이라 지표가 천장(0.95+)에 붙고 결함 1장이 F2 를 약 0.05 움직인다. 합성 효과 같은 작은 차이를 가리려면
test 결함을 늘려야 한다. `baseline_split.csv` 는 건드리지 않고 `splits/synth_test{N}_split.csv` 를 따로 만든다.

- 유형(type1~5)별 균등: 결함 test = N장 → 유형당 N/5 장. 정상 test 는 baseline 비율(결함 19 : 정상 36)을 유지한다.
- N=30 ⊂ 40 ⊂ 50 이 포함 관계(nested)가 되도록 시드 고정 순열의 앞에서부터 뽑는다.
- 나머지는 모두 train(→ 실험에서 80/20 으로 train/val). 합성 소스·대상은 이 train 안에서만 뽑는다(누수 통제).
"""
import os
import csv

import numpy as np

from src.common.paths import SPLITS_DIR
from src.supervised.data import list_records, CLASS_NAMES

TYPES = CLASS_NAMES[1:]
GOOD_PER_DEFECT = 36 / 19          # baseline test 의 정상:결함 비율


def split_csv(n_defect):
    return os.path.join(SPLITS_DIR, f'synth_test{n_defect}_split.csv')


def make_split(n_defect, seed=42, write=True):
    """(train_records, test_records). 레코드: split/class/filename/path/mask. 유형당 n_defect/5 장이 test 결함."""
    assert n_defect % len(TYPES) == 0, '결함 수는 5의 배수여야 한다(유형 균등)'
    per_type = n_defect // len(TYPES)
    n_good = round(n_defect * GOOD_PER_DEFECT)
    allrec = list_records('train') + list_records('test')
    rng = np.random.default_rng(seed)
    test_ids = set()
    for c in CLASS_NAMES:
        rs = sorted([r for r in allrec if r['class'] == c], key=lambda r: r['filename'])
        perm = rng.permutation(len(rs))                      # 순열은 유형마다 고정 → N 이 커져도 앞부분이 그대로(nested)
        n = n_good if c == 'good' else per_type
        assert n <= len(rs) // 2, f'{c}: test {n}장이 너무 많다(총 {len(rs)})'
        test_ids |= {id(rs[i]) for i in perm[:n]}
    train = [dict(r, split='train', _orig=r['split']) for r in allrec if id(r) not in test_ids]
    test = [dict(r, split='test', _orig=r['split']) for r in allrec if id(r) in test_ids]
    if write:
        with open(split_csv(n_defect), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['split', 'class', 'original_split', 'filename'])
            for r in train + test:
                w.writerow([r['split'], r['class'], r['_orig'], r['filename']])
    return train, test
