# Screw Defect Anomaly Detection → MES

나사(Screw) 결함을 **비지도 이상탐지(anomaly detection)** 로 검출·국소화하고,
검사 결과를 **MES(제조실행시스템)** 와 연동하는 것을 목표로 하는 프로젝트.

> 기존 졸업논문(지도학습 이진 분류)의 고도화 버전. 데이터 부족·미세 결함(Type1/Type4) 미검출 문제를
> 이상탐지로 해결하고, 결함 **위치를 픽셀 단위로 정량 평가**한다.

## 왜 이상탐지인가

기존 방식은 good/bad 이진 분류라 결함 이미지(유형당 ~20장)를 학습에 써야 했고, Type1(나사끝)·Type4(나사산 측면)
같은 미세 결함이 잘 안 잡혔다. 이상탐지는 **정상(good) 이미지만 학습**하므로:

- "결함 유형당 20장" 데이터 부족 문제가 **학습 단계에서 사라짐**
- 결함 119장 **전부를 평가에 사용**
- ground-truth 마스크로 **결함 위치를 정량 평가**(Pixel AUROC / AUPRO)

![결함 위치 히트맵 — 정상 + 유형별](results/figures/heatmap_readme.png)

> 정상(good)은 차분하고, 결함은 그 위치만 뜨겁게 반응한다(괄호=이상점수, 임계값 0.513).
> good/결함을 **공통 색 스케일**로 칠하고 배경을 억제해 대조를 정직하게 표현했으며, 각 유형의
> 표본은 **GT 마스크와의 농도비**로 자동 선별했다. 생성: `python build_heatmap_grid.py`.

## 데이터셋

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) 의 `screw` 카테고리
(good 361 / 결함 119, 5개 유형 + 픽셀 마스크).

| 유형 | MVTec 명칭 | 설명 |
|------|-----------|------|
| Type1 | manipulated_front | 나사끝 휘어짐/끊어짐 |
| Type2 | scratch_head | 나사머리 깨짐 |
| Type3 | scratch_neck | 나사목 깨짐 |
| Type4 | thread_side | 나사산 측면 |
| Type5 | thread_top | 나사산 상단 |

다운로드 후 아래처럼 배치 (레포에는 포함하지 않음):
```
data/screw/
 ├ train/good/                       # 320
 ├ test/good/  test/<defect>/        # 41 + 119
 └ ground_truth/<defect>/            # 119 masks
```

## 환경 설치

```bash
# 1) CUDA 12.1 빌드 torch 먼저
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# 2) 나머지
pip install -r requirements.txt
```
> Windows에서 OMP 충돌(`libiomp5md.dll`) 시 환경변수 `KMP_DUPLICATE_LIB_OK=TRUE` 설정.
> 코드 상단에서 이미 처리하지만, 직접 실행 시 참고.

## 모델 비교

세 이상탐지 모델을 동일 데이터·지표로 비교(`run_model_comparison.py`) → **PatchCore가 검출·위치 양면 최선**.

| 모델 | Image AUROC | AUPRO(위치) | 비고 |
|------|-------------|-------------|------|
| **PatchCore** (WideResNet-50, layer2+3, coreset) | **0.96** | **0.966** | 주력 |
| EfficientAD (small, 학습형) | 0.85~0.93 | 0.935 | 해상도↑에 민감 |
| PaDiM (resnet18) | 0.76 | 0.908 | Recall 1.0이나 오경보 많음 |

상세: [results/model_compare/](results/model_compare/RESULTS.md)

## 실행 (연구 트랙)

```bash
python smoke_patchcore.py             # 스모크런 (파이프라인 1회 검증)
python run_resolution_sweep.py 0.01   # 해상도 256/320 스윕 (coreset 인자)
python run_model_comparison.py all 20 # PaDiM/EfficientAD 비교
```
결과·히트맵은 `outputs/`(원본, gitignore) / `results/`(커밋된 큐레이션본)에 저장.

## 평가 지표 & 결과

- **이미지 단위**: AUROC, F2(임계값 튜닝, FN 최소화), Recall, Precision
- **픽셀 단위(위치)**: Pixel AUROC, **AUPRO**
- **유형별**: Type1~5 검출률 + 위치 품질 → Type1/Type4 개선 정량화

핵심 결과 (PatchCore):

| 설정 | Image AUROC | AUPRO | Type4 위치 | 시간 |
|------|-------------|-------|-----------|------|
| 256, coreset 0.1 | 0.963 | 0.941 | 0.971 | 146s |
| **320, coreset 0.1** | 0.961 | **0.966** | **0.984** | 70분 |
| 320, coreset 0.01 | 0.964 | 0.956 | 0.978 | **9분** |

→ **해상도↑로 Type4(나사산 측면) 위치 0.971→0.984 개선**(가설 입증). coreset 0.01로 320을 70분→9분에
실용화(정확도 손실 미미). 상세: [results/sweep_cs01](results/sweep_cs01/RESULTS.md) · [sweep_cs001](results/sweep_cs001/RESULTS.md)

## MES PoC (`mes/`)

학습한 PatchCore를 **실제 생산 라인처럼 운용**하는 MES-lite 데모. 라인 시뮬레이터가 흘려보낸
이미지를 실시간 판정하고, 결과를 쌓아 관제 대시보드로 보여준다.

```
simulator        inference          storage        dashboard
test 이미지   →  PatchCore(.pt)  →  SQLite     →   Streamlit 관제
1장씩 흘림       good/bad+점수       기록           KPI·불량률 추이·히트맵·알람
(양품 비율조정)  +위치 히트맵
```

**실행** (두 터미널):
```bash
python -m mes.core.export_model                      # (최초 1회) ckpt → 배포용 .pt
python -m mes.simulator.line_feeder inf 1.0 0.9       # A: 라인 가동(양품 90%)
streamlit run mes/dashboard/app.py                    # B: 대시보드 → localhost:8501
```

- **검출 성능**: 임계값 0.513(F2 최적, FN 최소화) → **Recall 0.992, Precision 0.929**.
  "불량을 놓치느니 정상을 재검사한다" 방침이라 정상 오경보를 일부 허용(의도된 trade-off).
- **결함 유형(Type1~5) 자동분류는 제외**: PatchCore 히트맵은 위치는 잘 짚지만 hot point가 유형과
  무관하게 중앙으로 쏠려, 위치 기반 분류가 데이터상 무작위 수준(≈20%)임을 확인 → good/bad + 위치 히트맵만 제공.
- 모델 = coreset 0.01/320(빠른 추론). 상세: [mes/README.md](mes/README.md).

## 로드맵

- [x] 이상탐지 파이프라인 (PatchCore + 마스크 평가) · 평가 레이어 (AUPRO·유형별)
- [x] 해상도/coreset 스윕, EfficientAD/PaDiM 비교
- [x] 추론 서비스화 (이미지 → good/bad + 이상점수 + 위치 히트맵)
- [x] **MES-lite PoC**: 결과 DB + 관제 대시보드(생산현황·불량률 추이·히트맵 갤러리·실시간 알람) + 라인 시뮬레이션
- [ ] (선택) FastAPI 추론 서비스 분리

자세한 설계는 [PLAN.md](PLAN.md), MES 상세는 [mes/README.md](mes/README.md) 참고.
