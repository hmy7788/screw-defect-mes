"""추론 지연(latency)·FPS 벤치마크 — coreset 0.1 vs 0.01 (PatchCore, 320).

정확도(AUROC/AUPRO)는 스윕에서 쟀지만 '속도'는 한 번도 측정하지 않았다.
PatchCore는 추론 시 메모리뱅크와 k-NN 거리계산을 하므로, 메모리뱅크 크기(coreset)가
곧 지연으로 직결된다. 이 스크립트는 그걸 정량화한다 → docs/decisions.md '정확도 vs 속도' 완성.

측정 원칙(중요):
  - **warm-up**: 첫 추론은 CUDA 커널 컴파일로 비정상적으로 느림 → 앞 N장은 버린다.
  - **torch.cuda.synchronize()**: GPU는 비동기라 동기화 없이 시간을 재면 거짓(빠른) 숫자가 나온다.
  - **end-to-end**: .pt에 내장된 전처리(Resize+Normalize)+forward+후처리 전 구간을 잰다(배포 현실).
  - **p50/p95**: 평균만이 아니라 꼬리 지연도 본다(라인 반응성은 최악값이 중요).

실행: python experiments/anomaly/bench_latency.py            (기본: cs01/320 vs cs001/320)
      python experiments/anomaly/bench_latency.py all        (256 변형까지 포함)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRUST_REMOTE_CODE", "1")
import sys, glob, time, shutil, statistics
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # 레포 루트
from src.common.paths import REPO_ROOT, DATA_DIR, OUTPUTS_DIR, RESULTS_DIR

import torch
import pandas as pd
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.deploy import ExportType, TorchInferencer
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, Normalize

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

ROOT = REPO_ROOT
TEST_DIR = os.path.join(DATA_DIR, "screw", "test")
BENCH_ROOT = os.path.join(OUTPUTS_DIR, "_bench")   # 임시 .pt (gitignore)
RESULTS_DIR = os.path.join(RESULTS_DIR, "anomaly", "latency")

# (이름, ckpt 상대경로, coreset, 해상도)
VARIANTS = [
    ("cs01_320",  "outputs/sweep_cs01/res320/Patchcore/MVTecADFixed/screw/v0/weights/lightning/model.ckpt",  0.1,  320),
    ("cs001_320", "outputs/sweep_cs001/res320/Patchcore/MVTecADFixed/screw/v0/weights/lightning/model.ckpt", 0.01, 320),
]
VARIANTS_256 = [
    ("cs01_256",  "outputs/sweep_cs01/res256/Patchcore/MVTecADFixed/screw/v0/weights/lightning/model.ckpt",  0.1,  256),
    ("cs001_256", "outputs/sweep_cs001/res256/Patchcore/MVTecADFixed/screw/v0/weights/lightning/model.ckpt", 0.01, 256),
]

WARMUP = 10      # 버리는 예열 추론 수
N_MEASURE = 80   # 측정 추론 수


def make_pp(res):
    return PreProcessor(transform=Compose([
        Resize((res, res), antialias=True),
        Normalize(mean=MEAN, std=STD),
    ]))


def export_variant(name, ckpt_rel, coreset, res):
    """ckpt → 배포용 .pt (변형별 임시 폴더). 경로 반환."""
    ckpt = os.path.join(ROOT, ckpt_rel)
    if not os.path.exists(ckpt):
        print(f"  [건너뜀] ckpt 없음: {ckpt_rel}")
        return None
    export_root = os.path.join(BENCH_ROOT, name)
    model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                      coreset_sampling_ratio=coreset, pre_processor=make_pp(res))
    engine = Engine(accelerator="gpu", devices=1)
    engine.export(model=model, export_type=ExportType.TORCH,
                  ckpt_path=ckpt, export_root=export_root)
    return os.path.join(export_root, "weights", "torch", "model.pt")


def memory_bank_size(inferencer):
    """로드된 모델의 메모리뱅크 패치 수(행) 추출 — '왜 느린가'의 근거 숫자."""
    try:
        mb = inferencer.model.model.memory_bank
        return int(mb.shape[0])
    except Exception:
        return None


def bench_one(name, pt_path, image_paths):
    inf = TorchInferencer(pt_path, device="auto")
    n_mb = memory_bank_size(inf)

    # warm-up (버림)
    for i in range(WARMUP):
        inf.predict(image_paths[i % len(image_paths)])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 측정: 1장당 end-to-end 지연
    lat_ms = []
    for i in range(N_MEASURE):
        img = image_paths[i % len(image_paths)]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = inf.predict(img)
        _ = float(batch.pred_score)          # 결과 강제 동기화
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    mean = statistics.mean(lat_ms)
    p50 = statistics.median(lat_ms)
    p95 = sorted(lat_ms)[int(0.95 * len(lat_ms)) - 1]
    fps = 1000.0 / mean
    return {
        "variant": name,
        "mem_bank": n_mb,
        "mean_ms": round(mean, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "fps": round(fps, 1),
    }


def main():
    variants = VARIANTS + (VARIANTS_256 if (len(sys.argv) > 1 and sys.argv[1] == "all") else [])
    image_paths = sorted(glob.glob(os.path.join(TEST_DIR, "*", "*.png")))
    if len(image_paths) < WARMUP + 5:
        print(f"[오류] 테스트 이미지가 부족합니다: {len(image_paths)}장 ({TEST_DIR})")
        sys.exit(1)
    print(f"[설정] 측정 이미지 풀={len(image_paths)}장  warm-up={WARMUP}  측정={N_MEASURE}장/변형")
    print(f"[장비] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    rows = []
    for name, ckpt_rel, coreset, res in variants:
        print(f"== {name} (coreset={coreset}, res={res}) ==")
        pt = export_variant(name, ckpt_rel, coreset, res)
        if pt is None:
            continue
        row = bench_one(name, pt, image_paths)
        row["coreset"] = coreset
        row["res"] = res
        rows.append(row)
        print(f"  메모리뱅크={row['mem_bank']} 패치  |  "
              f"평균={row['mean_ms']}ms  p50={row['p50_ms']}ms  p95={row['p95_ms']}ms  "
              f"FPS={row['fps']}\n")

    if not rows:
        print("측정된 변형이 없습니다."); return

    df = pd.DataFrame(rows)[["variant", "coreset", "res", "mem_bank",
                             "mean_ms", "p50_ms", "p95_ms", "fps"]]
    print("=" * 64)
    print("지연/FPS 종합 비교")
    print("=" * 64)
    print(df.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "latency.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_csv}")

    # 임시 .pt 정리
    shutil.rmtree(BENCH_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
