"""모델 비교 스윕 — PaDiM / EfficientAD 를 256/320 해상도로 학습·평가.

PatchCore 결과(run_resolution_sweep.py)와 **같은 지표 체계**(collect_predictions/evaluate_all)로
비교한다. coreset 은 PatchCore 전용이라 여기엔 없고, 각 모델의 고유 설정(학습 배치·epoch)은
아래 MODELS 테이블에 둔다.

  - PaDiM       : 정상 특징의 가우시안 분포. fit 1회(epoch 무관). resnet18 백본. 빠름·오프라인.
  - EfficientAD : student-teacher 학습형. **학습 배치=1 강제**, 학습 epoch 필요,
                  첫 실행 시 imagenette2.tgz(~1.5GB) 자동 다운로드(인터넷 필요).

실행:
  python run_model_comparison.py padim              # PaDiM만 (빠름)
  python run_model_comparison.py efficient_ad 20    # EfficientAD만, 20 epoch
  python run_model_comparison.py all 20             # 둘 다
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
from anomalib.models import Padim, EfficientAd
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, Normalize

from src.ad_data import build_screw_datamodule
from src.ad_eval import collect_predictions, evaluate_all

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RES_LIST = [256, 320]
OUT = "outputs/model_compare"
os.makedirs(OUT, exist_ok=True)

# 명령줄 인자: 어떤 모델을 / EfficientAD epoch 몇 번
WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"        # padim | efficient_ad | all
EFFAD_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def make_pp(res, normalize=True):
    """전처리기. EfficientAD는 정규화를 내부에서 하므로 normalize=False(Resize만) 필요.
    (anomalib: 'Transforms for EfficientAd should not contain Normalize' 에러 회피)"""
    tfs = [Resize((res, res), antialias=True)]
    if normalize:
        tfs.append(Normalize(mean=MEAN, std=STD))
    return PreProcessor(transform=Compose(tfs))


# 모델별 설정: 빌더 / 학습 epoch / 해상도별 학습 배치
# (EfficientAD는 train_batch_size=1 이 강제됨 — anomalib에서 ValueError)
MODELS = {
    "padim": dict(
        build=lambda res: Padim(pre_processor=make_pp(res)),   # backbone=resnet18 기본
        epochs=1,
        batch={256: 8, 320: 4},
    ),
    "efficient_ad": dict(
        build=lambda res: EfficientAd(model_size="small",
                                      pre_processor=make_pp(res, normalize=False)),
        epochs=EFFAD_EPOCHS,
        batch={256: 1, 320: 1},
    ),
}

if WHICH == "all":
    targets = list(MODELS.keys())
elif WHICH in MODELS:
    targets = [WHICH]
else:
    print(f"알 수 없는 모델: {WHICH} (padim | efficient_ad | all 중 하나)")
    sys.exit(1)

print(f"[설정] 대상={targets}  해상도={RES_LIST}  EfficientAD_epochs={EFFAD_EPOCHS}  출력={OUT}")

summaries = []
for name in targets:
    cfg = MODELS[name]
    for res in RES_LIST:
        print("\n" + "=" * 64)
        print(f"[{name} / 해상도 {res}] 학습·평가 시작 (epochs={cfg['epochs']}, batch={cfg['batch'][res]})")
        print("=" * 64)
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.manual_seed(42)

            bs = cfg["batch"][res]
            dm = build_screw_datamodule(root="data", category="screw",
                                        train_batch_size=bs, eval_batch_size=bs,
                                        num_workers=0, seed=42)
            dm.setup()
            model = cfg["build"](res)
            engine = Engine(default_root_dir=f"{OUT}/{name}_res{res}",
                            accelerator="gpu", devices=1, max_epochs=cfg["epochs"])

            t0 = time.time()
            engine.fit(model=model, datamodule=dm)
            data = collect_predictions(engine, model, dm.test_dataloader())
            summary, table = evaluate_all(data)
            dt = time.time() - t0

            summary = {"model": name, "res": res, **summary, "sec": round(dt, 1)}
            summaries.append(summary)
            table.to_csv(f"{OUT}/per_type_{name}_res{res}.csv", index=False, encoding="utf-8-sig")

            print(f"\n[{name} / {res}] 완료 ({dt:.1f}s)")
            print(f"  image_AUROC={summary['image_AUROC']:.4f}  F2={summary['F2']:.4f}  "
                  f"Recall={summary['Recall']:.4f}  Precision={summary['Precision']:.4f}")
            print(f"  pixel_AUROC={summary['pixel_AUROC']:.4f}  AUPRO={summary['AUPRO']:.4f}")
            print(f"  혼동행렬: TP={summary['TP']} FP={summary['FP']} FN={summary['FN']} TN={summary['TN']}")
            print("  유형별:")
            print(table.to_string(index=False))
        except RuntimeError as e:
            print(f"[{name} / {res}] 실패(메모리 등): {str(e)[:160]}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[{name} / {res}] 예외: {type(e).__name__}: {str(e)[:200]}")
            continue

print("\n" + "=" * 64)
print("모델 비교 종합")
print("=" * 64)
if summaries:
    df = pd.DataFrame(summaries)[
        ["model", "res", "image_AUROC", "F2", "Recall", "Precision", "pixel_AUROC", "AUPRO", "FN", "sec"]
    ]
    print(df.to_string(index=False))
    df.to_csv(f"{OUT}/summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT}/summary.csv, per_type_*.csv")
else:
    print("성공한 실행이 없습니다.")
