# HANDOFF — 새 세션 인계 문서

> 이 파일은 이전 세션(원본 `screw_defect` 폴더에서 진행)에서 이 레포로 작업을 넘기며 작성한 인계 문서입니다.
> 새 세션을 이 폴더에서 시작했다면 먼저 이 문서를 읽고 이어가세요. 상세 설계는 [PLAN.md](PLAN.md).

## 한 줄 요약

나사 결함을 **비지도 이상탐지(PatchCore/anomalib)** 로 검출·국소화하고, 최종적으로 **MES 연동(PoC)** 까지 가는 프로젝트. 기존 졸업논문(지도학습 이진 분류)의 고도화 버전.

## 지금까지 완료된 것

- ✅ 데이터: MVTec AD `screw` 원본(+ground_truth 마스크)을 **이 레포 `data/screw`에 실제 보유**
  (옛 `screw_defect`에서 이동 완료, 정션 아님). `outputs/`도 이 레포로 이동됨. 둘 다 gitignore 대상.
- ✅ 환경: conda env **`screw_defect`**(전역) / torch 2.5.1+cu121 / RTX 2070 SUPER / **anomalib 2.5.0**.
- ✅ 파이프라인: PatchCore 스모크런 통과. 평가 레이어(`src/ad_eval.py`) 완성 — AUPRO·유형별 검출률·위치.
- ✅ 새 Private 레포 생성·push: `github.com/hmy7788/screw-defect-mes`.

## 실행 방법 (이 레포 루트에서)

```bash
# 반드시 screw_defect conda 환경 + KMP 플래그
python smoke_patchcore.py        # 1회 파이프라인 검증
python run_resolution_sweep.py   # 해상도 256/320/512 비교 + 유형별 표
```
- 실행 파이썬: `C:\Users\dlsgh\.conda\envs\screw_defect\python.exe`
- **OMP 충돌 회피**: 스크립트 상단에서 `KMP_DUPLICATE_LIB_OK=TRUE` 처리. 직접 실행 시 환경변수로도 줄 것.
- Windows 콘솔 cp949 → 스크립트에서 stdout을 utf-8로 reconfigure 함(em-dash 등 주의).

## ⚠️ 알려진 함정 (해결됨)

- **anomalib `Split` enum 버그(Python 3.11)**: `samples.split == Split.TRAIN` 가 pandas에서 0건 매칭 →
  데이터 0개 로드. 우회: `src/ad_data.py::MVTecADFixed`가 split을 문자열로 전달. 검증됨(train 320/test 160).
- **AUPRO**: anomalib 2.5.0의 `AUPRO`는 Batch 필드 방식이라 원시 텐서로 실패 → 순수 베이스
  `from anomalib.metrics.aupro import _AUPRO` 사용(`ad_eval.py`에 반영됨).
- **anomalib 설치 시 torch 보호**: `torch==2.5.1+cu121` 제약 걸고 설치할 것(안 그러면 CPU torch로 덮어씀).

## 해상도 스윕 결과 (PatchCore, coreset 0.1) — 완료

| 지표 | 256 | 320 | 변화 |
|------|-----|-----|------|
| Image AUROC | 0.9627 | 0.9611 | ≈ |
| F2 / Recall | 0.974 / **1.0 (FN=0)** | 0.970 / 0.983 (**FN=2**) | 검출 소폭↓ |
| Precision | 0.881 | 0.921 | ↑ |
| Pixel AUROC | 0.9897 | 0.9929 | ↑ |
| AUPRO | 0.9409 | **0.9663** | ↑ (+0.025) |
| 시간 | 146초 | **4212초(70분)** | 29배↑ |

유형별 위치(pixel AUROC):

| Type | 256 | 320 |
|------|-----|-----|
| Type1 | 0.982 | 0.984 |
| Type2 | 0.992 | 0.997 |
| Type3 | 0.998 | 0.998 |
| **Type4** | **0.971** | **0.984 (+0.014, 최대 개선)** |
| Type5 | 0.993 | 0.995 |

**핵심 결론**:
- 해상도↑ → **Type4(나사산 측면) 위치가 가장 크게 개선**(0.971→0.984). 위치 지표 전반↑. 가설 입증.
- 단, 320에서 이미지 검출은 FN 2개 발생(Type3·Type4 각 1) — 위치는↑인데 임계값 판정은 흔들림.
- **현실 제약**: 320이 70분(coreset 병목), **512는 8GB에서 OOM**. coreset 0.1로 고해상도는 비현실적.
- **다음**: coreset를 **0.01~0.02로 낮춰** 320/512 재실행(시간 수십배↓, 정확도 유지), 320 FN 원인(임계값) 분석.

결과 CSV: 이 레포 `outputs/sweep_cs01/`(전체) + `results/sweep_cs01/`(커밋된 큐레이션본: CSV·Type4 히트맵).

## coreset 스윕 (0.1 vs 0.01) — 완료

coreset 0.1을 0.01로 낮춰 320 비용을 낮추는 후속 실험. 결과 CSV: `results/sweep_cs001/`.

| 지표 (320) | coreset 0.1 | coreset 0.01 | 변화 |
|-----------|------------|-------------|------|
| 시간 | 70분 | **9분(538s)** | **7.8배↓** |
| Image AUROC | 0.9611 | 0.9639 | 소폭↑ |
| Recall / FN | 0.983 / FN=2 | 0.992 / **FN=1** | 검출 동급+ |
| AUPRO | **0.9663** | 0.9560 | −0.010 |
| Pixel AUROC | 0.9929 | 0.9904 | −0.0025 |
| Type4 위치 | **0.984** | 0.978 | −0.007 |

**결론**: 0.01은 고해상도를 실용화(70→9분)하고 정확도 손실 미미. 단 **최고 AUPRO는 여전히 0.1**.
운용 가이드 = 정밀 우선 0.1@320 / 처리량 우선 0.01@320. **512는 coreset와 무관하게 추론 메모리(~7.5GB)
한계로 OOM 지속 → 미진행으로 마무리.** 스크립트는 `python run_resolution_sweep.py 0.01`처럼 인자로 coreset 조정.

## 진행 중 / 미완

- ✅ **해상도 스윕(256/320/512)**: 완료. 256/320 산출, 512는 OOM(8GB). 결과는 위 "해상도 스윕 결과" 참고.
  핵심: Type4 위치 0.971→0.984 개선(coreset 0.1).
- ✅ **coreset 스윕(0.1 vs 0.01)**: 완료(위 표). 0.01로 320=9분, 손실 미미. 512 OOM은 해상도 한계로 확정.
- ✅ **EfficientAD / PaDiM 비교**: 완료. 결과 `results/model_compare/`, 스크립트 `run_model_comparison.py`.
  3모델×256/320. 검출 image AUROC: **PatchCore 0.96 ≫ EfficientAD 0.85~0.93 > PaDiM 0.76**(PaDiM은
  Recall 1.0이나 FP~40 오경보). 위치 AUPRO도 PatchCore 0.966 > EffAD 0.935 > PaDiM 0.908.
  → **PatchCore가 검출·위치 양면 최선**(주력 모델 근거 확보). coreset은 PatchCore 전용이라 비교축은
  해상도. EfficientAD는 학습형(batch=1 강제, Normalize 전처리 금지, imagenette 1.5GB 다운로드,
  20 epoch 간이학습이라 다소 부족 — epoch↑ 시 개선 여지).
- 🔶 **MES PoC** (진행 중, `mes/`):
  - ✅ **Phase 1 추론 코어 완료**: `mes/core/export_model.py`(ckpt→배포용 .pt) +
    `mes/core/inference.py`(`ScrewInferencer.predict_one` → good/bad·이상점수·히트맵).
    - 모델 = coreset 0.01/320 ckpt를 `ExportType.TORCH`로 export(전처리·후처리 내장). `TorchInferencer` 사용.
    - good/bad 임계값 0.513(F2 최적): **Recall 0.992, FP 9/41**. 히트맵=이상도 비례 alpha 오버레이.
    - **유형(Type1~5) 자동분류는 제외 확정**: 위치 기반 분류가 데이터상 무작위 수준(≈20%, 유형별
      축위치 t 0.46~0.55로 완전 겹침)이라 불가. anomaly map은 위치는 잘 짚지만 hot point가 중앙 쏠림.
  - ⬜ Phase 2 storage(SQLite) / 3 simulator(라인) / 4 dashboard(Streamlit: 불량률 추이+히트맵 갤러리) / 5 api(선택)
  - 입력은 기존 데이터 스트리밍(라인 시뮬레이션). 대시보드=Streamlit 단일앱 채택.

## 사용자 결정 사항(확정)

- 방식: 비지도 이상탐지 세그멘테이션 / 이진 유지 + 유형별 분석
- 마스크: MVTec 원본 사용(학습 X, val X, **test 결함 119장 위치 평가에만**)
- 데이터: 480장 고정 / 속도는 신경 안 씀(정확도·위치 우선)
- 레포: 별도 Private 레포로 이관 / 입력: 기존 데이터 스트리밍

## 코드 구조

```
src/ad_data.py    # MVTec datamodule (enum 버그 우회)
src/anomaly.py    # 모델 빌드(PatchCore/EfficientAD/PaDiM) + 학습/평가 래퍼
src/ad_eval.py    # AUPRO·유형별 검출률·위치 IoU·임계값 튜닝(F2)
src/utils.py      # set_seed(cudnn.deterministic), make_run_dir
smoke_patchcore.py / run_resolution_sweep.py  # 실험 스크립트
```

메모리(`dev-environment`, `anomaly-detection-enhancement`)도 이 레포 프로젝트 폴더로 복사해 둠.
