# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

MVTec AD `screw` 데이터셋으로 나사 결함을 검출·국소화하고, 그 모델을 MES(제조실행시스템) 형태로 운용하는 프로젝트.
졸업논문의 **지도학습 baseline**에서 출발해 **비지도 이상탐지(PatchCore, anomalib)** 로 고도화했고, 두 가지 MES(Python PoC / C# 산업형)로 운용한다.
문서·주석·커밋 메시지는 모두 **한국어**로 작성한다(커밋은 `feat(scope): ...` 컨벤션). 전체 그림은 `docs/overview.md`, 결정 근거는 `docs/decisions.md`, MES 운용 부분은 `docs/mes.md`(루트 README는 실험 중심).

```
src/common      경로(paths.py = 모든 경로의 기준)·set_seed 등 공용
src/supervised  baseline(분류기 학습·Grad-CAM 위치평가)   ← experiments/supervised/
src/anomaly     이상탐지(anomalib 래퍼·평가)             ← experiments/anomaly/
mes/            Python MES PoC (SQLite + 시뮬레이터 + Streamlit)
csharp/         C# 산업형 MES (WPF + ONNX + SQL Server/EF Core)
data/screw      MVTec 원본 한 벌 (gitignore)    splits/baseline_split.csv  baseline 분할 매니페스트 (추적)
results/        커밋하는 결과 (supervised/, anomaly/)    outputs/  학습 원본 산출물 (gitignore)
```

## 환경 / 실행

- Python은 conda env **`screw_defect`** 하나로 baseline·이상탐지·Python MES가 모두 동작한다(검증: torch 2.11.0+cu128, anomalib 2.5.1). 설치는 **CUDA torch를 먼저** 깔고 `pip install -r requirements.txt`(안 그러면 anomalib이 CPU torch로 덮어씀).
- **경로는 항상 `src/common/paths.py`(`REPO_ROOT`, `MVTEC_DIR`, `OUTPUTS_DIR`, `RESULTS_DIR`, …)에서 가져온다.** cwd 상대경로(`./data/...`, `"outputs/..."`)를 새로 쓰지 말 것 — 스크립트는 어느 위치에서 실행해도 동작해야 한다. `experiments/*` 스크립트는 `__file__` 기준으로 레포 루트를 `sys.path`에 넣고 `from src....` 로 import 한다.
- 데이터: `data/screw/{train/good, test/good, test/<defect>, ground_truth/<defect>}` (MVTec 원본, gitignore). `outputs/`(학습 원본 ckpt·산출물), `*.ckpt`, `*.pth`, `*.onnx`, `mes/runtime/*`, `csharp/runtime/` 도 gitignore — 로컬에만 존재하므로 없으면 스크립트로 재생성한다.
- Windows 이슈: 스크립트 상단에서 `KMP_DUPLICATE_LIB_OK=TRUE`(OMP 충돌) 설정 및 stdout/stderr를 utf-8로 `reconfigure` 한다(cp949 콘솔). 새 스크립트도 같은 패턴. DataLoader는 `num_workers=0`.
- **테스트 스위트·린터는 없다.** 검증은 셀프테스트/파리티 체크/결과 재현으로 한다.

### baseline (지도학습)

```bash
python experiments/supervised/localization_eval.py   # ResNet-18 Grad-CAM 위치평가 (outputs/supervised/best_model.pth 필요; 결과 results/supervised/localization.csv 와 일치해야 함)
jupyter lab experiments/supervised                   # cnn_model_comparison / mobilenet_repeated_holdout 노트북 (노트북은 그 폴더에서 실행)
```

### 이상탐지

```bash
python experiments/anomaly/smoke_patchcore.py              # 파이프라인 1회 검증
python experiments/anomaly/run_resolution_sweep.py 0.01    # 해상도 256/320 스윕; 인자 = coreset 비율 (0.1 | 0.01)
python experiments/anomaly/run_model_comparison.py all 20  # PaDiM / EfficientAD 비교 (20 = epoch)
python experiments/anomaly/bench_latency.py [all]          # 추론 latency·FPS (warm-up + cuda.synchronize + p50/p95)
python experiments/anomaly/build_patchcore_summary.py      # results/anomaly/sweep_*·latency → patchcore_all.csv
python experiments/anomaly/build_heatmap_grid.py           # README용 히트맵 그리드 (results/anomaly/figures/)
```

### README용 그림 (`docs/images/`)

```bash
python experiments/figures/build_charts.py           # 차트 5종 (results CSV에서; 색 규칙은 스크립트 docstring)
python experiments/figures/extract_baseline_figures.py  # baseline 그림: docs/thesis PDF에서 추출 + outputs/supervised 위치평가 그림 축소
python experiments/figures/build_dataset_samples.py  # 데이터 샘플
python experiments/figures/build_comparison.py       # baseline vs PatchCore 비교 (mes/runtime의 .pt + outputs/supervised/best_model.pth 필요)
```

비교 그림의 표본은 baseline test의 유형별 첫 장(고정 규칙)이다 — PatchCore에 유리한 장면만 고르지 말 것.

원본 산출물은 `outputs/`(gitignore), 커밋하는 큐레이션본은 `results/anomaly/`.

### Python MES PoC (두 터미널)

```bash
python -m mes.core.export_model                    # (최초 1회) outputs의 ckpt → 배포용 .pt
python -m mes.core.inference                       # 셀프테스트(단일 추론 + 임계값 재보정)
python -m mes.simulator.line_feeder inf 1.0 0.9    # A: 라인 가동 (장수|inf, 간격(초), 양품 비율)
streamlit run mes/dashboard/app.py                 # B: 대시보드 localhost:8501
```

`mes/api/server.py`(FastAPI)는 **TODO 스텁**(미구현)이다.

### C# MES (`csharp/`, .NET 8 + SQL Server LocalDB 필요)

```bash
python -m mes.core.export_onnx                         # (레포 루트) ckpt → csharp/model/model.onnx + meta.json + parity_samples.json
cd csharp
dotnet build ScrewMes.sln
dotnet run --project ScrewMes.ParityCheck -c Release   # C# 추론 == Python 점수 검증 (36장, 최대 |Δ|≈1e-4)
dotnet run --project ScrewMes.App -c Release           # WPF 실행 (첫 실행 시 EF Migrate로 스키마 자동 생성)
```

DB 연결문자열은 `ScrewMes.Data/DbConfig.cs`(LocalDB 하드코딩). 앱은 `AppPaths.FindRepoRoot`가 `data/screw/test`를 찾을 때까지 상위 폴더를 거슬러 올라가 레포 루트를 정한다.

## 아키텍처 (여러 파일을 읽어야 보이는 큰 그림)

### baseline 데이터 접근 — 매니페스트 (`src/supervised/data.py`)
baseline의 train 425 / test 55 분할은 폴더 복사본이 아니라 **`splits/baseline_split.csv`**(split·class·filename·image·mask)가 정의하고, 이미지·마스크는 모두 `data/screw`를 가리킨다. 데이터 접근은 `list_records(split, classes)` / `list_class` / `class_of(path)` / `image_path` 로만 한다 — **경로 문자열 파싱(`'type1' in path`)으로 유형을 판별하지 말 것**(`class_of` 사용). 레코드는 (class 순서, filename) 정렬이며, 이 순서가 시드 고정된 분할·5-Fold 재현성의 전제이므로 바꾸지 말 것. 매니페스트나 정렬을 건드리면 `docs/decisions.md` §11의 검증(분할 목록·라벨·fold 인덱스·바이트 일치, `localization_eval.py` 지표 재현)을 다시 확인한다.

### 이상탐지 (`src/anomaly/`)
- `ad_data.py` — `MVTecADFixed`: anomalib `Split` enum 버그(Py3.11에서 데이터 0건) 우회를 위해 split을 **문자열**로 전달. 데이터가 0건 로드되면 이 경로부터 의심.
- `models.py` — `build_model(name)`(patchcore | efficient_ad | padim) + `run_experiment`. PatchCore는 epoch 학습이 아닌 메모리뱅크 구성.
- `ad_eval.py` — AUPRO·유형별 검출률·F2 임계값 튜닝. AUPRO는 anomalib의 공개 `AUPRO`가 아니라 **`anomalib.metrics.aupro._AUPRO`** 를 써야 한다(원시 텐서 입력 가능). baseline `loc_eval.py`도 같은 이유로 `_AUPRO` 사용.
- 비교축: PatchCore는 해상도×coreset, EfficientAD/PaDiM은 해상도뿐(coreset은 PatchCore 전용). 512 해상도는 8GB GPU에서 OOM(추론 거리행렬 ~7.5GB) — 시도하지 말 것.

### 단일 모델의 두 가지 배포 경로 (핵심)
같은 PatchCore ckpt(coreset 0.01 / 320, `outputs/sweep_cs001/res320/.../model.ckpt` — 경로는 `run_resolution_sweep.py` 출력 규칙과 결합돼 있어 `mes/config.py`와 함께 유지)를 두 형태로 export한다.
- **Torch(.pt)** → `mes/core/inference.py::ScrewInferencer.predict_one` (Python MES).
- **ONNX** → `csharp/ScrewMes.Inference/OnnxScrewInferencer` (C# MES).

두 경로 모두 anomalib export가 **전처리(Resize+Normalize)·후처리(점수 정규화)를 그래프에 내장**하므로 입력은 원본 RGB `/255` 텐서면 된다(수동 정규화 금지, `csharp/model/meta.json`이 입력 형태의 정답). **good/bad 판정은 모델 내장 `pred_label`이 아니라 `config.ANOMALY_THRESHOLD = 0.513`(F2 최적, FN 최소화; Recall 0.992)** 으로 직접 한다 — C# 쪽도 같은 값을 사용하므로 임계값을 바꾸면 양쪽을 함께 갱신하고 `parity_samples.json`을 재생성해야 한다.

### Python MES 데이터 흐름
`simulator/line_feeder` → `core/inference.predict_one`(판정+점수+히트맵 PNG) → `storage/repository`(SQLite, 스키마 `storage/schema.sql`) → `dashboard/app.py`(Streamlit이 DB를 폴링). 시뮬레이터는 test 셋(결함 75%)을 `good_ratio`로 복원추출해 양품 비율을 현실화한다.

### C# MES 레이어 (`csharp/ScrewMes.sln`)
`Core`(도메인 + `IScrewInferencer`/`I*Repository`, 의존성 0) ← `Inference`(ONNX Runtime + jet 히트맵) / `Data`(EF Core 코드퍼스트, SQL Server, 마이그레이션) ← `Simulator`(LineFeeder) ← `App`(WPF MVVM, `MainViewModel`). `ParityCheck`는 별도 콘솔. 추적성 모델은 `WorkOrder` 1—N `Lot` 1—N `Inspection`. 엔티티를 바꾸면 `ScrewMes.Data`에서 EF 마이그레이션을 추가해야 한다.

## 반드시 지킬 것 / 함정

- **결함 유형(Type1~5) 자동분류는 의도적으로 제외**했다(위치 기반 분류가 무작위 수준 ≈20%). 모델 출력은 good/bad + 점수 + 히트맵뿐이다. C#의 `Inspection.DefectType`은 모델 산출이 아니라 시뮬레이터가 소스 폴더명(GT)으로 주입한 값이며, good 소스를 bad로 판정하면 `정상(오탐)`으로 집계된다 — 이 "정직 표기"를 훼손하지 말 것.
- 화면 불량률이 ~25%로 보이는 것은 FN 최소화 임계값(0.513)의 대가인 정상 오경보(FP 9/41)로, 버그가 아니라 프로젝트 방침상의 trade-off다.
- baseline과 이상탐지 지표는 split이 달라(test 55 vs 160장, 위치평가 19 vs 119장) **직접 비교하지 않는다**(문서에도 그렇게 명시).
- **anomalib은 데이터 경로에 비ASCII 문자가 있으면 거부한다**(`validate_path`; 사용자 폴더가 한글인 PC에서 발생). `build_screw_datamodule`이 상대경로로 바꿔 넘기므로 `data/screw`를 anomalib에 절대경로로 직접 넘기지 말 것.
- `anomalib`을 (재)설치할 때 CUDA torch를 먼저 설치하지 않으면 CPU torch로 교체된다.
- `mes/core/export_onnx.py` docstring이 언급하는 `_onnx_probe.py`는 레포에 없다(ONNX 입력 형태 실측용 일회성 진단 스크립트였음).
