"""PatchCore 4개 변형(cs01/cs001 × 256/320)의 정확도+속도를 한 CSV로 통합.

흩어진 결과를 변형 기준으로 join:
  - 정확도: results/anomaly/sweep_cs01|cs001/summary.csv  (image/pixel AUROC, F2, Recall, AUPRO, 학습시간)
  - 속도:   results/anomaly/latency/latency.csv           (메모리뱅크, latency p50/p95, FPS)

출력: results/anomaly/patchcore_all.csv  (변형당 1행, 정확도+속도 한눈에)
재현: python experiments/anomaly/build_patchcore_summary.py
"""
import os
import pandas as pd

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # 레포 루트
from src.common.paths import RESULTS_DIR

R = lambda *p: os.path.join(RESULTS_DIR, "anomaly", *p)


def _clean(df):
    """정렬용 공백(컬럼명·문자열 셀 끝)을 제거한다 — pandas 버전에 따라 skipinitialspace 만으로는 남아 merge 키가 어긋남."""
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    return df


def load_acc(coreset_tag, coreset_val):
    """sweep summary(res별 행) → variant 키 부여."""
    # skipinitialspace: 가독성 위해 콤마 뒤 정렬 공백이 있어도 컬럼명/값을 깨끗이 읽는다.
    df = pd.read_csv(R(f"sweep_{coreset_tag}", "summary.csv"), skipinitialspace=True)
    df = _clean(df)
    df = df.rename(columns={"sec": "train_sec"})
    df["coreset"] = coreset_val
    df["variant"] = [f"{coreset_tag}_{int(r)}" for r in df["res"]]
    return df


def main():
    acc = pd.concat([load_acc("cs01", 0.1), load_acc("cs001", 0.01)], ignore_index=True)
    lat = pd.read_csv(R("latency", "latency.csv"), skipinitialspace=True)
    lat = _clean(lat)

    # variant 기준 병합 (coreset/res는 양쪽에 있으니 latency쪽 중복 컬럼 제거)
    lat = lat.drop(columns=[c for c in ("coreset", "res") if c in lat.columns])
    df = acc.merge(lat, on="variant", how="outer")

    # 정확도 소수 4자리로 정리
    acc_cols = ["image_AUROC", "F2", "Recall", "Precision", "pixel_AUROC", "AUPRO"]
    for c in acc_cols:
        df[c] = df[c].round(4)

    # 보기 좋은 순서로 컬럼 정렬
    cols = ["variant", "coreset", "res",
            "image_AUROC", "F2", "Recall", "Precision", "pixel_AUROC", "AUPRO", "FN",
            "mem_bank", "mean_ms", "p50_ms", "p95_ms", "fps", "train_sec"]
    df = df[cols].sort_values(["coreset", "res"], ascending=[False, True]).reset_index(drop=True)

    out = R("patchcore_all.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
