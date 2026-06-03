"""해상도 스윕 — PatchCore를 256/320/512로 학습·평가해 Type1/Type4 개선을 비교.

각 해상도: PatchCore(WideResNet-50, layer2+3, coreset 10%) 학습 후
이미지/픽셀/유형별 지표 산출. 고해상도 OOM 시 해당 해상도만 건너뜀.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, gc, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, Normalize

from src.ad_data import build_screw_datamodule
from src.ad_eval import collect_predictions, evaluate_all

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RES_LIST = [256, 320, 512]
BATCH = {256: 8, 320: 4, 512: 2}
OUT = "outputs/sweep"
os.makedirs(OUT, exist_ok=True)


def make_pp(res):
    return PreProcessor(transform=Compose([
        Resize((res, res), antialias=True),
        Normalize(mean=MEAN, std=STD),
    ]))


summaries = []
for res in RES_LIST:
    print("\n" + "=" * 64)
    print(f"[해상도 {res}] PatchCore 학습/평가 시작")
    print("=" * 64)
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.manual_seed(42)

        bs = BATCH[res]
        dm = build_screw_datamodule(root="data", category="screw",
                                    train_batch_size=bs, eval_batch_size=bs,
                                    num_workers=0, seed=42)
        dm.setup()
        model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                          coreset_sampling_ratio=0.1, pre_processor=make_pp(res))
        engine = Engine(default_root_dir=f"{OUT}/res{res}",
                        accelerator="gpu", devices=1, max_epochs=1)

        t0 = time.time()
        engine.fit(model=model, datamodule=dm)
        data = collect_predictions(engine, model, dm.test_dataloader())
        summary, table = evaluate_all(data)
        dt = time.time() - t0

        summary = {"res": res, **summary, "sec": round(dt, 1)}
        summaries.append(summary)
        table.to_csv(f"{OUT}/per_type_res{res}.csv", index=False, encoding="utf-8-sig")

        print(f"\n[해상도 {res}] 완료 ({dt:.1f}s)")
        print(f"  image_AUROC={summary['image_AUROC']:.4f}  F2={summary['F2']:.4f}  "
              f"Recall={summary['Recall']:.4f}  Precision={summary['Precision']:.4f}")
        print(f"  pixel_AUROC={summary['pixel_AUROC']:.4f}  AUPRO={summary['AUPRO']:.4f}")
        print(f"  혼동행렬: TP={summary['TP']} FP={summary['FP']} FN={summary['FN']} TN={summary['TN']}")
        print("  유형별:")
        print(table.to_string(index=False))
    except RuntimeError as e:
        print(f"[해상도 {res}] 실패(메모리 등): {str(e)[:160]}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue
    except Exception as e:
        print(f"[해상도 {res}] 예외: {type(e).__name__}: {str(e)[:160]}")
        continue

print("\n" + "=" * 64)
print("해상도 스윕 종합 비교")
print("=" * 64)
if summaries:
    df = pd.DataFrame(summaries)[
        ["res", "image_AUROC", "F2", "Recall", "Precision", "pixel_AUROC", "AUPRO", "FN", "sec"]
    ]
    print(df.to_string(index=False))
    df.to_csv(f"{OUT}/summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT}/summary.csv, per_type_res*.csv")
else:
    print("성공한 해상도가 없습니다.")
