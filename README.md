<div align="center">

# Screw Defect Detection → MES
### 지도학습 baseline에서 비지도 이상탐지(PatchCore)로, 지도학습 재개선(합성·정렬)과 MES 운용까지

나사 결함의 유무와 위치를 찾고, 학습한 모델을 생산 현장의 MES로 운용한 프로젝트

개인 프로젝트 · 졸업논문 고도화 (2026.02 ~ 2026.06)

<img src="results/anomaly/figures/heatmap_portfolio.png" width="820" alt="정상과 결함 유형별 이상 점수 맵">

정상은 차분하고 결함은 그 위치만 뜨겁게 반응 (괄호 = 이상 점수, 판정 임계값 0.513)

</div>

## 프로젝트 요약

- 졸업논문의 지도학습 분류(ResNet-18)는 결함 유형당 20장으로 학습해야 했고, 분류기라 결함의 위치를 짚지 못함 (Type4 위치 AUPRO 0.45)
- 정상(good) 이미지만 학습하는 비지도 이상탐지(PatchCore)로 전환해 결함 119장 전부를 평가에 쓰고, 픽셀 마스크로 위치를 정량 평가
- 해상도, coreset, 모델 3축으로 비교 실험해 정확도와 속도의 트레이드오프를 정량화
- 지도학습 baseline의 데이터 부족과 낮은 해상도를 되짚어, 결함 합성(Cut & Paste)과 입력 정렬(나사를 세워 해상도 확보)로 ResNet-18을 다시 개선하고 효과를 통제 실험으로 검증
- 평가 표본이 작아(결함 19장) 차이가 가려지는 문제를 확인하고, test 결함을 30/40/50장으로 키운 분할에서 결론을 다시 검증하는 과정에서 중간 결론을 2번 정정
- 학습한 모델을 Python MES와 C# 산업형 MES 두 형태로 운용 (같은 모델을 .pt와 ONNX로 배포)
- 실행 방법: [docs/overview.md](docs/overview.md#6-실행-명령-레포-루트에서)

| 항목 | 결과 |
|---|---|
| 데이터셋 | MVTec AD screw 480장 (정상 361, 결함 119, 5유형, 픽셀 마스크) |
| baseline | ResNet-18 Test F2 0.9896 (논문 보고 값), 위치는 Type4 Pixel AUROC 0.65 |
| 검출 성능 | PatchCore Image AUROC 0.96, 임계값 0.513에서 결함 119장 중 118장 검출 (Recall 0.992) |
| 위치 성능 | AUPRO 0.966, Pixel AUROC 0.993 (coreset 0.1, 해상도 320), Type4 위치 0.971 → 0.984 (256 → 320) |
| 처리 속도 | 28 FPS (p50 35ms, coreset 0.01, 해상도 320, 측정 당시 RTX 2070 SUPER) |
| 지도학습 개선 | 입력 정렬로 정상 오경보 21.6 → 1.8장, Image AUROC 0.868 → 0.988, 위치 AUPRO 0.310 → 0.819 (test 결함 50장 / 정상 95장, 시드 5개 평균) |
| 합성 효과 | 정렬 입력에서 합성 비율 1.0 → F2 0.901 → 0.959 (p=0.01), AUPRO 0.819 → 0.920 (p=0.01), 합성 없을 때 성능이 낮을수록 효과가 큼 |
| MES | Python(Streamlit, SQLite) + C#(WPF, ONNX, SQL Server), C#과 Python 점수 일치 36/36 |

※ 지도학습 개선 행은 평가 분할(test 145장)이 baseline(55장)·PatchCore(160장)와 달라 다른 행과 직접 비교하지 않음

## 목차

1. [개발 배경](#1-개발-배경)
2. [프로젝트 구성](#2-프로젝트-구성)
3. [데이터셋](#3-데이터셋)
4. [실험 요약](#4-실험-요약)
5. [기술 스택](#5-기술-스택)
6. [프로젝트 정보](#6-프로젝트-정보)
7. [회고](#7-회고)
8. [관련 문서](#8-관련-문서)

## 1. 개발 배경

<img src="docs/images/chart_baseline_vs_patchcore.png" width="820" alt="baseline과 PatchCore의 유형별 위치 정확도">

- 데이터 부족: 지도학습 baseline은 결함 이미지 100장(유형당 20장)을 학습에 써야 해 분산이 크고, 평가에 남는 결함은 19장뿐
- 위치를 못 짚음: 분류기는 위치를 학습하지 않아 Grad-CAM에 의존, 분류 성능(F2 0.9896)이 높아도 Type4(나사산 측면)는 Pixel AUROC 0.65, AUPRO 0.45
- 데이터 재발견: 논문 데이터 480장이 MVTec AD screw와 바이트 단위로 동일함을 해시로 확인 → 정상만 학습하는 이상탐지로 바꾸면 결함 119장 전부를 평가에 쓰고 위치를 정량 평가 가능

## 2. 프로젝트 구성

```mermaid
flowchart LR
    D[("MVTec screw<br/>480장 + 픽셀 마스크")]
    D --> B["baseline<br/>ResNet-18, MobileNet-V2, VGG-16<br/>(지도학습)"]
    D --> P["이상탐지<br/>PatchCore, EfficientAD, PaDiM<br/>(비지도학습)"]
    B --> S["지도학습 개선<br/>결함 합성 + 입력 정렬<br/>(ResNet-18)"]
    B --> E["평가<br/>Image AUROC, F2, Recall<br/>Pixel AUROC, AUPRO, 유형별"]
    S --> E
    P --> E
    P --> CK(["체크포인트"])
    CK -->|"export .pt"| PY["Python MES-lite<br/>Streamlit, SQLite"]
    CK -->|"export ONNX"| CS["C# 산업형 MES<br/>WPF, SQL Server"]
```

| 트랙 | 하는 일 | 핵심 결과 | 상세 |
|---|---|---|---|
| baseline | 지도학습 이진분류 3종 비교, Grad-CAM 위치평가 | ResNet-18 F2 0.9896, Type4 위치 0.65 | [docs/baseline.md](docs/baseline.md) |
| PatchCore | 정상만 학습, 모델·해상도·coreset 비교 | AUPRO 0.966, Type4 위치 0.984 | [docs/patchcore.md](docs/patchcore.md) |
| 지도학습 개선 | 결함 합성(Cut & Paste) + 나사 정렬 입력, 통제 실험 | 정렬로 오경보 21.6 → 1.8장, 합성으로 AUPRO 0.82 → 0.92 | [docs/synthetic.md](docs/synthetic.md) |
| MES | 모델을 검사 시스템에 연결 (Python, C#) | 28 FPS, 파리티 36/36 | [docs/mes.md](docs/mes.md) |

- 코드: `src/supervised`(baseline), `src/anomaly`(이상탐지), 실행 스크립트는 `experiments/`, 모든 경로는 `src/common/paths.py` 기준
- 전체 구조와 환경, 실행 명령: [docs/overview.md](docs/overview.md)

## 3. 데이터셋

<img src="docs/images/dataset_samples.png" width="820" alt="데이터셋 샘플">

- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) screw 카테고리 (1024×1024, 픽셀 마스크 제공, CC BY-NC-SA 4.0 — 비상업적 연구용, 출처 표기)
- 결함 5유형: 나사끝(Type1), 나사머리(Type2), 나사목(Type3), 나사산 측면(Type4), 나사산 상단(Type5)
- 이상탐지는 MVTec 표준 분할(학습 정상 320, 평가 정상 41 + 결함 119), baseline은 논문 분할(학습 425, 평가 55)을 `splits/baseline_split.csv`로 정의하고 이미지는 `data/screw` 한 벌만 사용

## 4. 실험 요약

### 4-1. baseline: ResNet-18, MobileNet-V2, VGG-16

<img src="docs/images/chart_baseline_models.png" width="820" alt="baseline 3개 모델 비교">

- 클래스 가중치 Grid Search + Stratified 5-Fold CV로 학습, ResNet-18이 모든 지표에서 최고 (Test F2 0.9896, 논문 보고 값)
- VGG-16은 학습 손실이 거의 줄지 않아 성능이 낮고, MobileNet-V2는 5-seed 반복에서 평균 F2 0.818 ± 0.044
- Grad-CAM 위치평가에서 Type4는 Pixel AUROC 0.65, AUPRO 0.45 — 이상탐지로 전환한 동기
- 상세: [docs/baseline.md](docs/baseline.md)

### 4-2. PatchCore 이상탐지

<img src="docs/images/compare_type4.png" width="820" alt="Type4에서 baseline과 PatchCore 비교">

- 모델: PatchCore가 EfficientAD, PaDiM보다 검출(Image AUROC 0.96)과 위치(AUPRO 0.966) 모두 최고
- 해상도: 256 → 320에서 Type4 위치 0.971 → 0.984로 가장 크게 개선
- coreset: 0.1 → 0.01로 학습 7.8배, 추론 2배 빨라지고 AUPRO 손실은 0.010 (MES에는 0.01 채택)
- 같은 이미지 비교: Type4 test 3장에서 PatchCore는 실제 결함 위치에 반응, baseline은 결함이 아닌 나사머리 쪽을 칠함 (단 PatchCore도 정상 오경보와 배경 오반응이 있음)
- 상세: [docs/patchcore.md](docs/patchcore.md)

### 4-3. 지도학습 개선: 결함 합성과 입력 정렬

**한 줄 요약**: 결함 사진이 너무 적고 너무 작게 줄여 넣는 문제를 풀려고, ① 결함을 오려 정상 나사에 붙여 학습 사진을 늘리고(합성) ② 나사를 똑바로 세워 크게 보여준(정렬) 뒤, 각각이 성능을 올리는지 같은 조건에서 비교한 실험

#### 왜 했나
- 결함 사진이 유형당 20장뿐이라 모델이 배울 예시가 적음
- 사진 한 장이 1024px인데 모델에는 224px로 줄여 넣어서, 약 40px짜리 작은 결함이 10px 정도로 뭉개짐

#### 무엇을 했나

**① 합성 (Cut & Paste)** — 실제 결함 사진에서 결함 부분만 오려, 결함 없는 정상 나사의 **같은 위치**에 붙여 "가짜 결함 사진"을 만듦

<img src="docs/images/synthetic_samples.png" width="520" alt="결함 합성 before/after">

유형별 첫 행(고정 규칙)의 예: 왼쪽 정상 나사 → 가운데 실제 결함(빨간 선이 결함 영역) → 오른쪽 합성 결과
- 위치는 "나사 머리에서 끝까지 몇 % 지점, 옆으로 얼마"라는 좌표로 맞춤 (나사마다 누워 있는 각도가 달라서 화면 좌표를 그대로 쓰면 허공에 붙음)
- 붙인 경계가 티 나지 않게 섞고(`seamlessClone`), 결함이 나사 몸통 밖에 걸치면 버림
- 오려 쓰는 결함은 **학습용 사진에서만** 뽑음. 평가용 사진의 결함을 쓰면 정답을 미리 보여주는 셈이라서(누수)

**합성 비율이란?** 정상 사진 수에 대해 가짜 결함 사진을 몇 장 더 넣느냐
- 예) 학습용 정상 사진이 약 213장일 때: 비율 0.25 → 약 53장 추가, 0.5 → 약 106장, 1.0 → 약 213장(정상과 1:1)
- 0은 합성 없음(기준선). 결함 유형 5가지에 똑같이 나눠 넣음
- 이 실험은 "얼마나 넣어야 좋은가"를 보려고 0 / 0.25 / 0.5 / 1.0을 모두 비교

**② 입력 정렬** — 나사를 회전시켜 항상 머리가 위로 오게 세운 뒤, 나사만 담은 세로로 긴 사진(128×384)으로 잘라 모델에 넣음
- 효과: 나사가 화면을 꽉 채워 결함이 크게 보이고, 나사가 누운 각도가 제각각인 문제도 사라짐

<img src="docs/images/aligned_input.png" width="820" alt="입력 정렬 전후">

위: 기존 입력(전체를 축소) / 아래: 정렬 입력

#### 어떻게 비교했나
- 입력 2종(기존 / 정렬) × 합성 비율 4종(0, 0.25, 0.5, 1.0)을 같은 설정으로 학습
- **평가는 진짜 사진만**: 합성 사진은 평가에 한 장도 쓰지 않음. 평가용 결함을 19장에서 30 / 40 / 50장으로 늘려서도 확인 (19장이면 한 장이 점수를 크게 흔들어 차이가 가려지기 때문)
- 같은 설정을 **무작위 시작값(시드)만 바꿔 5번** 반복해 평균을 냄. 한 번 돌려서 운이 좋았던 것을 가려내기 위해서

#### 점수 읽는 법
| 지표 | 뜻 | 좋은 방향 |
|---|---|---|
| F2 | 결함을 놓치지 않는 쪽에 더 무게를 둔 종합 점수 (불량 출하가 재검사보다 치명적이라서) | 높을수록 |
| 정상 오경보(FP) | 정상 나사를 불량이라고 잘못 판정한 장수 | 적을수록 |
| Image AUROC | 임계값과 상관없이 "정상과 결함을 얼마나 잘 구분하는가" (0.5 = 찍기, 1.0 = 완벽) | 높을수록 |
| AUPRO | 모델이 "여기가 결함"이라고 짚은 위치가 실제 결함 영역과 얼마나 겹치는가 | 높을수록 |
| p 값 | 차이가 우연일 확률. 0.05 보다 작으면 "우연으로 보기 어렵다"(5번 반복의 평균 차이 기준) | 작을수록 |

#### 결과 (평가용 결함 50장 / 정상 95장, 5번 반복 평균)

<img src="docs/images/synthetic_ratio_chart.png" width="820" alt="입력과 합성 비율별 성능">

그래프: 가로축이 합성 비율, 회색이 기존 입력, 빨강이 정렬 입력 (세로 막대는 5번 반복의 편차)

1. **정렬이 가장 큰 개선 요인**
   - 합성 없이도 정상 오경보 21.6장 → 1.8장, 정상/결함 구분(Image AUROC) 0.868 → 0.988, 위치 겹침(AUPRO) 0.310 → 0.819
   - 평가용 결함 30 / 40 / 50장 어느 쪽으로 평가해도 같은 방향
2. **합성은 정렬 위에서도 도움이 됨**
   - 합성 비율 1.0일 때 F2 0.901 → 0.959 (p=0.01), 위치 겹침 0.819 → 0.920 (p=0.01)
   - 합성 없이도 이미 점수가 높은 경우(평가용 결함 30장)에는 차이가 없었음. **합성 없을 때 성능이 낮을수록 효과가 큼**
3. 비율은 "많을수록 좋다"로 정리되지 않음: 0.25 / 0.5 / 1.0 사이의 차이는 평가 크기마다 달랐고, 이 실험으로는 최적 비율을 정하지 못함

#### 모델이 어디를 보는가 (Grad-CAM)

<img src="docs/images/synthetic_gradcam_align.png" width="820" alt="정렬 입력 모델의 합성 비율별 Grad-CAM">

붉을수록 "여기가 결함 같다"고 본 곳, 두 번째 열이 실제 결함 위치(정답). 정렬 입력 모델 1개(시드 0), 유형별 첫 장으로 고정해 골랐음
- 대체로 실제 결함 근처를 붉게 봄 (type1, type3)
- 아쉬운 점도 그대로 기록: type2는 합성 1.0 모델에서 결함 대신 나사 아래쪽을 봄, 정상 나사에서 합성 0.5~1.0 모델이 머리 쪽을 약하게 의심함(불량 확률 0.03~0.17)

#### 과정에서 결론이 바뀐 일
- 처음에 "정렬 + 합성 0.25가 최선"이라고 했다가, 비율을 다 돌려 보고 **철회**
- 평가용 결함 19장일 때 "정렬 위에서는 합성이 도움이 안 된다"로 읽었는데, 평가를 50장으로 키우니 **뒤집힘**. 효과가 없던 게 아니라 표본이 작아 구분이 안 됐던 것

#### 한계
- 평가용 30 / 40 / 50장 분할은 서로 포함 관계이고 시드도 같아서 "세 번 독립적으로 재현"한 것이 아님
- 여러 지표를 한꺼번에 비교하면서 p 값 보정을 하지 않았음
- 정렬이 좋아진 이유가 "정렬" 때문인지 "해상도가 커져서"인지는 분리하지 못함 (정렬 없이 해상도만 키운 대조 실험이 필요)
- 정렬 전처리가 사진 한 장당 약 86 ms(CPU)라 처리 속도(FPS)에는 불리함

- 상세 표와 근거: [docs/synthetic.md](docs/synthetic.md), 결정 기록: [docs/decisions.md §14](docs/decisions.md)

### 4-4. MES 운용

<img src="docs/images/demo_dashboard.gif" width="640" alt="Python MES 시연 화면">

- Python MES-lite: 라인 시뮬레이터 → 추론 → SQLite → Streamlit 관제 (불량률·이상점수 추이, 결함 위치 히트맵)
- C# 산업형 MES: WPF(MVVM) + EF Core + SQL Server, ONNX Runtime으로 .NET에서 직접 추론, 작업지시 → Lot → 검사 추적성
- 같은 체크포인트를 .pt와 ONNX로 export, C#과 Python 점수 일치(36/36)
- 상세: [docs/mes.md](docs/mes.md)

## 5. 기술 스택

| 구분 | 기술 |
|---|---|
| 이상탐지 | anomalib 2.5, PyTorch (PatchCore, EfficientAD, PaDiM) |
| baseline | PyTorch, torchvision (ResNet-18, MobileNet-V2, VGG-16), pytorch-grad-cam |
| 평가 | scikit-learn, Pixel AUROC, AUPRO (`anomalib.metrics.aupro`) |
| 모델 배포 | TorchInferencer(.pt), ONNX, ONNX Runtime |
| Python MES | Streamlit, SQLite, pandas |
| C# MES | .NET 8, WPF(MVVM, CommunityToolkit.Mvvm), EF Core 8, SQL Server LocalDB, LiveCharts2, ImageSharp |
| 환경 | Windows 11, conda, CUDA |

## 6. 프로젝트 정보

| 항목 | 내용 |
|---|---|
| 기간 | 2026.02 ~ 2026.06 |
| 형태 | 개인 프로젝트 (졸업논문 baseline 고도화) |
| 역할 | 데이터 분석, 모델링, 평가 설계, 비교 실험, Python/C# MES 구현 전 과정 |
| 졸업논문 | [딥러닝 기반 나사 결함 탐지 모델 성능 비교에 관한 연구](docs/thesis/딥러닝%20기반의%20나사%20결함%20탐지%20모델%20성능%20비교에%20관한%20연구.pdf) |

## 7. 회고

- 잘한 점
  - baseline의 한계를 수치(Type4 위치 0.65/0.45)로 확인한 뒤 이상탐지로 전환하고, 해상도, coreset, 모델 3축으로 비교해 선택 근거를 정량화
  - 결함 유형 자동분류를 시도하고, 무작위 수준(20%)임을 데이터로 확인한 뒤 제외
  - 같은 모델을 .pt와 ONNX 두 경로로 배포하고 C#과 Python 점수 일치(36/36)를 검증
  - 논문 데이터가 MVTec 원본과 동일함을 해시로 확인해 중복 데이터 377MB를 제거하고, 리팩터링 전후 분할과 지표가 동일함을 검증
  - 평가 표본이 작아 차이가 가려지는 것을 알아채고 test 결함을 19 → 30/40/50장으로 키워 재검증, 그 과정에서 이전 결론을 2번 정정
  - 합성 소스·대상을 학습분에서만 뽑고 합성 정의를 별도 매니페스트(`synthetic_split.csv`)로 분리해 누수와 baseline 재현성을 지킴
- 아쉬운 점
  - 정렬이 효과인지 해상도 때문인지 분리하지 못했고, 정렬 전처리가 느려(약 86 ms) 속도 면에서 불리
  - 평가 분할이 달라 지도학습 개선 트랙과 PatchCore를 같은 기준으로 비교하지 못함
  - 임계값 0.513은 test셋에서 F2를 최대화해 골랐기 때문에, 같은 데이터로 튜닝하고 평가한 Recall 0.992는 낙관적일 수 있음 (val 홀드아웃이 필요)
  - 정상 오경보가 큼: 재학습 모델은 정상 41장 중 15장을 불량으로 판정, 학습마다 편차(9~15장)가 있고 대시보드 불량률이 40%대로 보임
  - baseline과 PatchCore 비교는 split이 달라 엄밀하지 않음, baseline test는 결함이 19장뿐이라 같은 설정으로 재학습해도 F2가 0.885~0.990으로 흔들림 (논문 0.9896, 저장 모델 0.885, 재학습 0.904)
  - 속도는 측정 당시 환경(RTX 2070 SUPER) 수치이고 이 PC에서 재측정하지 않음, 시연은 MVTec test 이미지 시뮬레이션이며 실제 카메라 입력은 미검증
- 향후 계획
  - val 홀드아웃으로 임계값 선정, 시드 고정과 반복 평가로 신뢰구간 제시
  - 정상 오경보 감소: 앙상블, 후처리 검토 (지도학습 쪽 입력 정렬은 4-3에서 검증 완료, PatchCore 쪽 적용은 미검증)
  - 정렬 전처리를 축소 해상도에서 수행해 지연 감소, 비정렬 고해상도(Resize 448) 대조군으로 효과 분리, 지도학습 개선 모델과 PatchCore를 같은 분할에서 비교
  - 실제 카메라 연동, FastAPI 추론 서비스 분리, 결함 유형은 별도 지도학습 보조 모델로 분류

## 8. 관련 문서

| 문서 | 내용 |
|---|---|
| [docs/baseline.md](docs/baseline.md) | 지도학습 baseline 3종, 학습 방법, Grad-CAM 위치평가 |
| [docs/synthetic.md](docs/synthetic.md) | 결함 합성 + 입력 정렬 실험 (방법, 비율별 결과, test 확대 재평가, 한계) |
| [docs/patchcore.md](docs/patchcore.md) | PatchCore 이상탐지 실험 (모델 비교, 해상도, coreset, 속도, baseline 비교) |
| [docs/mes.md](docs/mes.md) | 학습한 모델을 운용하는 MES (Python, C#, ONNX 파리티) |
| [docs/overview.md](docs/overview.md) | 레포 구조, 데이터, 환경, 실행 명령, 진행 현황 |
| [docs/decisions.md](docs/decisions.md) | 결정과 문제 해결 기록 |
| [mes/README.md](mes/README.md), [csharp/README.md](csharp/README.md) | MES 코드 상세 |

## 참고

- MVTec AD: Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019
- PatchCore: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022
- anomalib: https://github.com/open-edge-platform/anomalib
