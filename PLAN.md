# 나사 결함 탐지 고도화 계획 — 이상탐지 세그멘테이션 전환

## Context (배경)

현재 프로젝트는 **이진 분류(good/bad)** 방식이다. Type1~5 결함을 모두 `label=1`로 병합해
ImageNet pretrained CNN(ResNet-18/MobileNet-V2/VGG-16)을 파인튜닝하고, 위치는 Grad-CAM으로 사후 시각화만 한다.

확인된 두 가지 한계:
1. **데이터 부족** — 결함이 유형당 train 20장(전체 119장)뿐. 학습 데이터가 극소량이라 분산이 크고
   Val-Test 성능 역전이 발생.
2. **Type1/Type4 미탐지** — 나사끝 휘어짐(Type1)·나사산 측면 깨짐(Type4)은 미세·기하학적 결함인데,
   이진 분류는 유형 구분 없이 학습되고 위치를 학습하지 않아 이 유형들이 잘 안 잡힘.

**핵심 통찰:** 이 데이터셋은 **MVTec AD의 screw 카테고리와 정확히 동일**하다
(good 361장 = MVTec train good 320 + test good 41, 결함 119장 = manipulated_front/scratch_head/
scratch_neck/thread_side/thread_top, 픽셀 단위 ground-truth 마스크 제공).
→ **비지도 이상탐지(anomaly detection)** 로 전환하면:
- good 이미지만 학습 → "결함 유형당 20장" 데이터 부족 문제가 **학습 단계에서 사라짐**.
- 결함 119장 **전부가 평가 데이터**로 회수됨(현재는 100장이 학습에 묶여 있음).
- 픽셀 마스크로 **결함 위치를 정량 평가**(IoU/AUPRO) 가능 → Type1/Type4 위치 정확도 측정.

**사용자 확정 사항:** 이상탐지 세그멘테이션 / 이진 유지 + 유형별 분석 / MVTec 원본 마스크 사용 / 데이터 480장 고정.

**목표 산출물:** 기존 지도학습 분류(baseline)와 신규 비지도 이상탐지(제안)를 비교하여,
이상탐지가 데이터 부족·Type1/Type4 미탐지·위치 정량화를 동시에 해결함을 보이는 고도화 연구.

---

## 전체 구조: 2-트랙 비교

| 트랙 | 방식 | 학습 데이터 | 역할 |
|------|------|-------------|------|
| **A. Baseline (기존)** | 지도학습 이진 분류 (ResNet-18 등) | good 325 + bad 100 | 비교 기준선. 코드 그대로 유지, 재실행만. |
| **B. 제안 (신규)** | 비지도 이상탐지 + 세그멘테이션 | **good만** | 본 고도화의 핵심. 위치 정량 평가. |

논문 서사: "데이터 부족·미세결함 환경에서 지도학습 이진 분류 대비 이상탐지의 우위" 비교 연구.

---

## Q1. 데이터 분할 (트랙 B 기준)

이상탐지는 결함을 학습에 쓰지 않으므로 분할 철학이 완전히 달라진다.

- **Train (good only):** `data/train/good` 325장 → 메모리뱅크/학습용.
  - 이 중 **30~40장을 threshold 보정용 val(good)** 로 홀드아웃(임계값 누수 방지).
- **Test/Eval:** `data/test/good` 36장(정상) **+ 결함 119장 전부**(train/test의 type1~5 모두 합침,
  학습에 안 쓰였으므로 전량 평가 가능) = 정상 36 + 결함 119.
  - 결함 119장 각각에 MVTec ground-truth 마스크를 매칭(아래 "선행작업" 참조).
- 유형 라벨(type1~5)은 폴더명으로 유지 → **유형별 분석** 그대로 가능.

> 트랙 A(baseline)는 기존 `full_train_test_split` / `custom_Stratified_K_Fold`(`src/dataset.py`)를 그대로 사용.

---

## Q2. 어떤 모델

속도 무관, 정확도·위치정밀도 우선 → MVTec screw에서 검증된 SOTA 이상탐지 모델 비교.

- **주력: PatchCore** — WideResNet-50 백본, layer2+layer3 특징 + coreset 메모리뱅크.
  MVTec screw 이미지 AUROC ~0.99, 픽셀 위치 우수. 학습 epoch 없음(특징 추출+coreset).
- **비교 1: EfficientAD** — student-teacher + autoencoder, 미세결함 위치에 강함.
- **비교 2: PaDiM 또는 Reverse Distillation** — 추가 baseline.
- **트랙 A 참조: ResNet-18 지도학습 분류**(기존 결과 F2=0.9896) — 비교표에 함께 게재.

**구현 라이브러리: `anomalib`** (오픈소스, PyTorch Lightning 기반).
- 이유: PatchCore/EfficientAD/PaDiM/Reverse Distillation 전부 내장 + **MVTec 폴더 포맷·마스크 로딩**
  + 이미지/픽셀 AUROC·**AUPRO** 메트릭 + 위치 히트맵 시각화를 모두 제공 → 재구현 불필요.
- 직접 재구현 대안(PatchCore ~200줄)도 가능하나, 마스크 정량평가·시각화 부담이 커 anomalib 권장.

---

## Q3. 성능 향상 포인트 (특히 Type1/Type4)

1. **입력 해상도 상향:** 현재 224 → **256, 320 비교**. Type1(끝 휘어짐)·Type4(나사산 측면)는
   미세·국소 결함이라 고해상도에서 위치·검출 모두 개선. (속도 무관하므로 320까지 실험)
2. **큰 백본 + 다층 특징:** WideResNet-50 layer2+3 (PatchCore 기본). ResNet-18 대비 표현력 ↑.
3. **PatchCore coreset 비율 스윕:** 1% / 10% / 25% — 큰 비율일수록 정확도↑(속도 무관하니 25%까지).
4. **회전 정합 처리:** 나사는 회전 방향이 다양(기존 코드도 RandomRotation 180 사용).
   이상탐지는 회전에 민감 → good 학습 시 회전 증강을 메모리뱅크에 포함하거나, 입력 정렬 검토.
   (단, 이상탐지는 과한 증강이 정상 분포를 흐릴 수 있어 회전 위주 최소 증강만)
5. **유형별 임계값/분석:** 단일 anomaly score 임계값 외에, Type1/Type4 점수 분포를 따로 분석해
   해당 유형 누락 구간을 진단(필요 시 유형 인지 후처리).
6. **앙상블(선택):** PatchCore + EfficientAD anomaly map 평균으로 위치 정밀도 보강.

---

## Q4. 학습 전략

- **트랙 B 학습:** good-only.
  - PatchCore: 메모리뱅크 적합(에폭 없음), coreset 서브샘플링.
  - EfficientAD: student-teacher steps 학습(anomalib 기본 스케줄).
- **임계값 선택:** good val 홀드아웃 + 정상/결함 점수 분포로 결정.
  **FN 최소화 기조 유지** → F2 최대 또는 목표 Recall(예: 1.0) 보장 임계값 선택(기존 철학 계승).
- **재현성:** 기존 `src/utils.py::set_seed()`(cudnn.deterministic 포함) 재사용, 시드 고정.
- **유형별 분석:** test 평가 시 Type1~5별 검출률(recall)·위치 IoU를 분리 집계.

### 평가 지표 (3계층)
1. **이진(이미지 단위):** Image AUROC, **F2-Score(주지표, 기존 계승)**, Recall, Precision.
2. **위치(픽셀 단위):** Pixel AUROC, **AUPRO**, 임계값에서 IoU/Dice. (MVTec 마스크 사용)
3. **유형별:** Type1~5 각각 검출률 + 위치 품질 → **Type1/Type4 개선 정량 입증**(핵심 결과표).

기존 FASI(속도 포함 복합지표)는 속도 무관 결정에 따라 **부차 지표로 격하**, AUPRO/IoU를 전면화.

---

## 선행 작업: MVTec ground-truth 마스크 확보·매칭 ⚠️

현재 `data/`에 마스크가 없고, 파일이 `type1/020.png`식으로 재명명되어 있어 매칭이 필요하다.

1. **MVTec AD screw 원본 취득** (공식 배포본: `screw/train/good`, `screw/test/<defect>/`,
   `screw/ground_truth/<defect>/000_mask.png`). 결함 119장에 1:1 마스크 존재.
2. **현재 파일 ↔ 원본 매칭:** 재명명으로 파일명이 다르므로 **이미지 해시(또는 픽셀 동일성) 매칭**으로
   현재 `data/.../typeN/*.png`를 MVTec 원본 결함 이미지에 대응시키고, 해당 마스크를 복사.
   - 유형 대응: Type1=manipulated_front, Type2=scratch_head, Type3=scratch_neck,
     Type4=thread_side, Type5=thread_top.
3. 권장 산출 구조(anomalib MVTec 포맷 호환):
   ```
   data_ad/screw/
     ├ train/good/                 # 325 (or 320)
     ├ test/good/                  # 36
     ├ test/type1..type5/          # 119
     └ ground_truth/type1..type5/  # 119 masks (xxx_mask.png)
   ```
> 매칭이 불가하면 대안: MVTec 원본 split을 그대로 사용(데이터 동일·480장 유지, 분할만 표준화).

---

## 구현 단계 (파일 단위)

신규 코드는 `src/`에 모듈 추가, 노트북은 신규 `anomaly_detection.ipynb`로 분리(기존 `main.ipynb` 보존).

1. **`prepare_mvtec_masks.py` (신규, 스크립트):** 위 "선행작업" — 원본 취득·해시 매칭·`data_ad/` 생성.
2. **`src/anomaly.py` (신규):** anomalib 래퍼 — 모델 빌드(PatchCore/EfficientAD/PaDiM),
   datamodule 구성(해상도·증강 파라미터화), 학습/예측, anomaly map 반환.
3. **`src/ad_eval.py` (신규):** 이미지 AUROC·F2·Recall, 픽셀 AUROC·AUPRO·IoU,
   **유형별(Type1~5) 검출률·IoU** 집계. 기존 F2 정의(`engine.py`)와 일관 유지.
4. **`src/ad_visualize.py` (신규 또는 `visualization.py` 확장):** anomaly heatmap + 마스크 오버레이,
   유형별 위치 비교 그리드(기존 Grad-CAM 그리드 스타일 계승).
5. **`anomaly_detection.ipynb` (신규):** 데이터 준비 → 모델별(해상도·coreset 스윕) 학습/평가
   → 비교표·유형별표·위치 시각화 → 트랙 A(ResNet-18) 결과와 합본 비교.
6. **재사용:** `src/utils.py::set_seed`, `make_run_dir`(outputs 저장 규약) 그대로 사용.
7. **`requirements`/README:** `anomalib`(+의존성) 추가, 고도화 방법론·결과 섹션 갱신.

---

## 검증 방법 (End-to-End)

1. **마스크 매칭 검증:** 119장 전부 마스크 1:1 대응 확인, 5장 샘플 오버레이로 마스크 정합 육안 확인.
2. **트랙 B 스모크런:** PatchCore, 해상도 256, coreset 10%로 1회 학습/평가 →
   Image AUROC·Pixel AUROC·AUPRO 출력 확인(anomalib 기준 screw AUROC ~0.95+ 나오면 정상).
3. **본 실험:** 모델 3종 × 해상도(256/320) × coreset 스윕 → 비교표 생성.
4. **핵심 검증(목표):** **유형별 표에서 Type1·Type4 검출률/IoU가 기존 대비 개선**되는지 확인.
5. **이진 지표 계승 확인:** F2·Recall이 baseline(ResNet-18 F2=0.9896, Recall=1.0) 대비 동등 이상,
   또는 위치 정량화라는 추가 가치를 입증.
6. **재현성:** 시드 고정 2회 재실행으로 지표 분산 확인.

---

## 리스크 / 주의

- **마스크 매칭**이 최대 리스크(파일 재명명). 해시 매칭 실패 시 MVTec 원본 split 사용으로 우회(데이터 불변).
- **anomalib 버전 호환:** PyTorch 2.10/cu128, RTX 5080(sm_120) 환경에서 동작 확인 필요. 안 되면
  PatchCore 경량 직접구현(WideResNet-50 특징 + faiss/torch coreset)으로 대체.
- **회전 민감성:** 나사 회전 다양성으로 정상 분포가 넓어 false positive 가능 → 회전 증강/정합으로 완화.
- **데이터 480 고정:** good-only 학습으로 결함 부족은 해소되나, 정상 다양성은 여전히 제한적 →
  good 학습셋 최대 활용(325장) 권장.
