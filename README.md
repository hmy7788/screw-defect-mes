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

## 모델

- **PatchCore** (WideResNet-50, layer2+3, coreset 메모리뱅크) — 주력
- EfficientAD / PaDiM — 비교 예정

## 실행

```bash
python smoke_patchcore.py        # 스모크런 (파이프라인 1회 검증)
python run_resolution_sweep.py   # 해상도 256/320/512 비교 + 유형별 분석
```
결과·결함 위치 히트맵은 `outputs/` 에 저장된다.

## 평가 지표

- **이미지 단위**: AUROC, F2(임계값 튜닝, FN 최소화), Recall, Precision
- **픽셀 단위(위치)**: Pixel AUROC, AUPRO
- **유형별**: Type1~5 검출률 + 위치 품질 → Type1/Type4 개선 정량화

### 현재 결과 (PatchCore, 256, coreset 10%)

| 지표 | 값 |
|------|-----|
| Image AUROC | 0.963 |
| F2 / Recall | 0.974 / **1.0 (FN=0)** |
| Pixel AUROC | 0.990 |
| AUPRO | 0.941 |

유형별 위치(pixel AUROC): Type3 0.998 · Type5 0.993 · Type2 0.992 · Type1 0.982 · **Type4 0.971(최저)**
→ 해상도 상향으로 Type4 개선을 실험 중.

## 로드맵

- [x] 이상탐지 파이프라인 (PatchCore + 마스크 평가)
- [x] 평가 레이어 (AUPRO · 유형별)
- [ ] 해상도/coreset 스윕, EfficientAD/PaDiM 비교
- [ ] 추론 서비스화 (이미지 → 판정 + 위치 + 유형, API)
- [ ] **MES-lite PoC**: 결과 DB + 관제 대시보드(생산현황·불량률·유형분포·실시간 알람) + 라인 시뮬레이션

자세한 설계는 [PLAN.md](PLAN.md) 참고.
