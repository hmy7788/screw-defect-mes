# 모델 비교 — PatchCore vs EfficientAD vs PaDiM (256/320)

MVTec `screw`, good-only 학습. 세 이상탐지 **방법**을 동일 데이터·동일 지표체계로 비교.
coreset 은 PatchCore 전용 손잡이라 비교 공통축은 **해상도(256/320)** 로 둠.

- 생성 스크립트: [`run_model_comparison.py`](../../run_model_comparison.py) (PaDiM/EfficientAD)
- PatchCore 수치는 coreset 0.1 결과([../sweep_cs01/](../sweep_cs01/))에서 가져옴.
- EfficientAD는 학습형이라 `model_size=small`, **20 epoch**(간이 학습)로 실행.

## 종합 ([summary.csv](summary.csv))

### 이미지 단위 검출 (불량 판별)

| 모델 | res | Image AUROC | F2 | Recall | Precision | FN | 시간(s) |
|------|-----|-------------|-----|--------|-----------|----|---------|
| **PatchCore** | 256 | **0.9627** | 0.9738 | 1.000 | 0.8815 | 0 | 145.8 |
| **PatchCore** | 320 | **0.9611** | 0.9701 | 0.9832 | 0.9213 | 2 | 4212.1 |
| EfficientAD | 256 | 0.8539 | 0.9497 | 0.9832 | 0.8357 | 2 | 1131.2 |
| EfficientAD | 320 | 0.9324 | 0.9477 | 0.9748 | 0.8529 | 3 | 1212.3 |
| PaDiM | 256 | 0.7590 | 0.9370 | 1.000 | 0.7484 | 0 | 30.3 |
| PaDiM | 320 | 0.7643 | 0.9355 | 1.000 | 0.7438 | 0 | 39.3 |

### 픽셀 단위 위치 (결함 위치 정확도)

| 모델 | res | Pixel AUROC | AUPRO |
|------|-----|-------------|-------|
| **PatchCore** | 320 | **0.9929** | **0.9663** |
| EfficientAD | 320 | 0.9818 | 0.9348 |
| PaDiM | 320 | 0.9760 | 0.9079 |

## 유형별 위치(pixel AUROC) @320 — Type4(나사산 측면) 중심

per_type CSV: [padim](per_type_padim_res320.csv) / [efficientad](per_type_efficient_ad_res320.csv)
(PatchCore는 [../sweep_cs01/per_type_res320.csv](../sweep_cs01/per_type_res320.csv))

| Type | PatchCore | EfficientAD | PaDiM |
|------|-----------|-------------|-------|
| Type1 | 0.9841 | 0.9319 | 0.9843 |
| Type2 | 0.9968 | 0.9958 | 0.9930 |
| Type3 | 0.9984 | 0.9989 | 0.9794 |
| **Type4** | **0.9843** | 0.9551 | 0.9453 |
| Type5 | 0.9950 | 0.9934 | 0.9759 |

## 결론

1. **검출 정밀도: PatchCore ≫ EfficientAD > PaDiM.** PatchCore Image AUROC 0.96으로 압도.
   PaDiM은 0.76(불량은 전부 잡지만 정상 ~40장 오경보, Precision 0.74), EfficientAD는 중간(0.85~0.93).
2. **해상도↑ 효과는 EfficientAD에서 가장 극적**: Image AUROC 256→320 = 0.854→0.932.
   위치 지표는 전 모델이 320에서 개선 → "고해상도가 미세결함에 유효"가 모델 무관하게 재확인.
3. **위치 품질도 PatchCore 우위**(AUPRO 0.966 > 0.935 > 0.908). Type4도 PatchCore가 최고(0.984).
4. **종합**: 이 데이터셋·환경에서 **PatchCore가 검출·위치 양면 최선** → 주력 모델 선택의 정량 근거.

> ⚠️ 공정성 주의: EfficientAD는 학습형이라 20 epoch면 다소 부족(논문은 수만 step). epoch 상향 시
> 개선 여지 있음 → "동일·간이 학습 조건 비교"로 해석할 것. PaDiM/PatchCore는 fit 1회(epoch 무관).

> 전체 히트맵·체크포인트는 용량 문제로 레포 미포함.
> 재현: `python run_model_comparison.py all 20` (PatchCore는 `run_resolution_sweep.py`).
