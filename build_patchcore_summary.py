"""PatchCore 4개 변형(cs01/cs001 × 256/320)의 정확도+속도를 한 CSV로 통합.

흩어진 결과를 변형 기준으로 join:
  - 정확도: results/sweep_cs01|cs001/summary.csv  (image/pixel AUROC, F2, Recall, AUPRO, 학습시간)
  - 속도:   results/latency/latency.csv           (메모리뱅크, latency p50/p95, FPS)

출력: results/patchcore_all.csv  (변형당 1행, 정확도+속도 한눈에)
재현: python build_patchcore_summary.py
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(ROOT, "results", *p)


def load_acc(coreset_tag, coreset_val):
    """sweep summary(res별 행) → variant 키 부여."""
    # skipinitialspace: 가독성 위해 콤마 뒤 정렬 공백이 있어도 컬럼명/값을 깨끗이 읽는다.
    df = pd.read_csv(R(f"sweep_{coreset_tag}", "summary.csv"), skipinitialspace=True)
    df = df.rename(columns={"sec": "train_sec"})
    df["coreset"] = coreset_val
    df["variant"] = [f"{coreset_tag}_{int(r)}" for r in df["res"]]
    return df


def main():
    acc = pd.concat([load_acc("cs01", 0.1), load_acc("cs001", 0.01)], ignore_index=True)
    lat = pd.read_csv(R("latency", "latency.csv"), skipinitialspace=True)

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
