# 프로젝트 개요

나사(Screw) 결함 검출 프로젝트의 전체 그림. 졸업논문(지도학습 이진분류)에서 출발해 **비지도 이상탐지**로 고도화하고,
그 모델을 **MES(제조실행시스템)** 두 가지 형태로 운용한다.

```
 지도학습 baseline ──한계──▶ 이상탐지(PatchCore) ──운용──▶ Python MES-lite (Streamlit/SQLite)
 (졸업논문, ResNet-18)       (정상만 학습, 위치 정량평가)      └▶ C# 산업형 MES (WPF/ONNX/SQL Server)
```

## 1. 왜 이 구조인가

졸업논문은 good/bad 이진분류(ResNet-18 등)였다. 두 가지 한계가 있었다.

1. **데이터 부족** — 결함이 유형당 train 20장뿐이라 학습 데이터가 극소량. 분산이 크고 Val-Test 성능 역전이 발생.
2. **미세 결함/위치** — Type1(나사끝)·Type4(나사산 측면)는 미세·국소 결함인데, 분류기는 위치를 학습하지 않아
   Grad-CAM 사후 시각화에 의존하고 Type4 위치는 거의 못 짚는다(Pixel AUROC 0.65, AUPRO 0.45; [baseline](baseline.md)).

**핵심 통찰**: 논문 데이터(480장)는 **MVTec AD `screw`와 바이트 단위로 동일**하다(good 361 = train 320 + test 41,
결함 119, 픽셀 GT 마스크 제공). → 정상만 학습하는 이상탐지로 바꾸면 "유형당 20장" 문제가 학습 단계에서 사라지고,
결함 119장 전부를 위치 평가에 쓸 수 있다. 상세 근거: [decisions.md](decisions.md).

## 2. 레포 구조

```
screw-defect-mes/
├ README.md  CLAUDE.md  requirements.txt
├ docs/                 이 문서들 (+ 졸업논문 PDF)
├ data/                 (gitignore) screw/ — MVTec 원본 한 벌만 둔다
├ splits/               baseline_split.csv — baseline 의 train/test 분할 매니페스트 (추적)
├ src/
│  ├ common/            paths.py(모든 경로의 기준), utils.py(set_seed, make_run_dir)
│  ├ supervised/        baseline: data(매니페스트 접근)·dataset·model·engine·visualization·loc_eval
│  └ anomaly/           이상탐지: ad_data(anomalib datamodule)·ad_eval(AUPRO·F2·유형별)·models
├ experiments/
│  ├ supervised/        노트북 2개 + localization_eval.py
│  └ anomaly/           smoke · 해상도/coreset 스윕 · 모델 비교 · latency 벤치 · 요약/히트맵 생성
├ results/              커밋하는 결과 — supervised/, anomaly/
├ outputs/              (gitignore) 학습 원본 산출물·ckpt
├ mes/                  Python MES PoC  → mes/README.md
└ csharp/               C# 산업형 MES   → csharp/README.md
```

핵심 규칙: **모든 경로는 `src/common/paths.py`(`REPO_ROOT`)를 기준**으로 한다. 스크립트는 레포 루트가 아니라 어디서 실행해도
동작해야 하며, `./data/...` 같은 cwd 상대경로를 새로 쓰지 않는다.

## 3. 데이터

| 위치 | 내용 |
|------|------|
| `data/screw/` | MVTec AD `screw` 원본: `train/good` 320 · `test/good` 41 · `test/<결함5종>` 119 · `ground_truth/<결함5종>` 119 마스크(1024×1024, {0,255}) |
| `splits/baseline_split.csv` | baseline 의 고정 분할(480행: split·class·filename·image·mask). 이미지·마스크는 `data/screw` 를 가리킨다 |

유형 대응: Type1 = manipulated_front(나사끝 휘어짐/끊어짐) · Type2 = scratch_head(나사머리) · Type3 = scratch_neck(나사목) ·
Type4 = thread_side(나사산 측면) · Type5 = thread_top(나사산 상단).

**baseline 분할**(논문과 동일): train 425 = good 325 + 유형당 20 · test 55 = good 36 + 4/4/5/3/3. 학습 시 Type1~5 는 모두 bad(1)로 병합.
과거에는 이 분할을 `new_k-fold_data/`·`new_k-fold_data_gt/`(약 377MB) 폴더 복사본으로 들고 있었으나 이미지가 `data/screw`와
100% 동일해, 매니페스트 한 파일로 대체했다(이미지 480장·마스크 119개 바이트 동일, 분할 순서·결과 재현 검증 완료 — [decisions.md](decisions.md) §11).

이상탐지 트랙은 anomalib 표준 MVTec 프로토콜을 그대로 쓴다: train = good 320, test = good 41 + 결함 119.

## 4. 환경

- Windows 11, conda env `screw_defect`. **baseline 과 이상탐지는 같은 환경 하나로 돌아간다.**
- 이 PC에서 검증된 버전: torch 2.11.0+cu128 · torchvision 0.26.0 · anomalib 2.5.1 · numpy 2.5.1 · pytorch-grad-cam 설치.
  (과거 환경: RTX 2070 SUPER / torch 2.5.1+cu121 / anomalib 2.5.0 — 당시 512 해상도 OOM 확인은 이 8GB GPU 기준.)
- 설치: **CUDA 빌드 torch 를 먼저** 설치한 뒤 `pip install -r requirements.txt` (그렇지 않으면 anomalib 이 CPU torch 로 덮어쓴다).
- **anomalib 은 데이터 경로에 ASCII 출력가능 문자만 허용**한다(`validate_path`). 사용자 폴더명이 한글이면 절대경로로 `data/screw` 를 넘길 때 `Path contains non-printable characters` 로 실패하므로 `build_screw_datamodule` 이 cwd 기준 상대경로로 바꿔 넘긴다.
- ONNX export(`mes.core.export_onnx`)에는 `onnx`·`onnxruntime` 이 필요하다(requirements.txt 에 포함).
- Windows 함정(코드에 이미 반영): `KMP_DUPLICATE_LIB_OK=TRUE`(OMP 충돌), stdout/stderr utf-8 reconfigure(cp949 콘솔), DataLoader `num_workers=0`.
- 그 외 함정은 [decisions.md](decisions.md) §2: anomalib `Split` enum 버그, AUPRO(`_AUPRO`) 사용, `TRUST_REMOTE_CODE=1`.

## 5. 평가 지표 (3계층)

1. **이미지 단위**: AUROC, **F2**(주지표 — Recall 가중, FN 최소화), Recall, Precision. baseline 은 FPS·FASI(F2+속도 복합)도 사용.
2. **픽셀 단위(위치)**: Pixel AUROC, **AUPRO**(결함 영역별 정규화라 큰 결함에 덜 편향). 결함 이미지만 대상으로 계산한다(good 포함 시 AUPRO 왜곡).
3. **유형별**: Type1~5 검출률 + 위치 품질 → Type1/Type4 개선 정량화.

## 6. 실행 명령 (레포 루트에서)

```bash
# --- baseline (지도학습) ---
python experiments/supervised/localization_eval.py     # Grad-CAM 위치평가 (outputs/supervised/best_model.pth 필요)
jupyter lab experiments/supervised                     # cnn_model_comparison / mobilenet_repeated_holdout 노트북

# --- 이상탐지 ---
python experiments/anomaly/smoke_patchcore.py          # 파이프라인 1회 검증
python experiments/anomaly/run_resolution_sweep.py 0.01  # 해상도 256/320 스윕 (인자 = coreset 비율)
python experiments/anomaly/run_model_comparison.py all 20  # PaDiM/EfficientAD (20 = epoch)
python experiments/anomaly/bench_latency.py [all]      # 추론 latency·FPS
python experiments/anomaly/build_patchcore_summary.py  # 정확도+속도 통합 CSV
python experiments/anomaly/build_heatmap_grid.py       # README 용 히트맵 그리드

# --- README 용 그림 (docs/images/) ---
python experiments/figures/build_charts.py             # 차트 5종 (results CSV 에서)
python experiments/figures/extract_baseline_figures.py # baseline 그림 (논문 PDF 에서 추출 + 위치평가 그림 축소)
python experiments/figures/build_dataset_samples.py    # 데이터 샘플
python experiments/figures/build_comparison.py         # baseline vs PatchCore 비교 (PatchCore .pt + baseline best_model.pth 필요)

# --- Python MES (두 터미널) ---
python -m mes.core.export_model                        # (최초 1회) ckpt → 배포용 .pt
python -m mes.simulator.line_feeder inf 1.0 0.9        # A: 라인 가동 (장수|inf, 간격 초, 양품 비율)
streamlit run mes/dashboard/app.py                     # B: 대시보드 localhost:8501

# --- C# MES (.NET 8 + SQL Server LocalDB) ---
python -m mes.core.export_onnx                         # ckpt → csharp/model/model.onnx + meta.json + parity_samples.json
dotnet run --project csharp/ScrewMes.ParityCheck -c Release
dotnet run --project csharp/ScrewMes.App -c Release
```

`outputs/` 의 ckpt(`outputs/sweep_cs001/res320/.../model.ckpt`)는 gitignore 라 로컬에만 있다. 없으면
`python experiments/anomaly/run_resolution_sweep.py 0.01` 로 재생성(약 9분)해야 MES(Python/C#)를 돌릴 수 있다.

## 7. 진행 현황

- [x] 졸업논문 baseline 편입 (`src/supervised`, `experiments/supervised`) + Grad-CAM 위치평가
- [x] 이상탐지 파이프라인(PatchCore + 마스크 평가), 평가 레이어(AUPRO·유형별)
- [x] 해상도/coreset 스윕, EfficientAD/PaDiM 비교, 추론 latency 벤치 → [patchcore.md](patchcore.md)
- [x] 추론 서비스화(이미지 → good/bad + 이상점수 + 위치 히트맵)
- [x] Python MES-lite: SQLite + 라인 시뮬레이터 + Streamlit 관제 → [mes.md](mes.md), [../mes/README.md](../mes/README.md)
- [x] C# 산업형 MES: WPF + ONNX 네이티브 + SQL Server/EF Core + 작업지시·Lot 추적성·Pareto → [../csharp/README.md](../csharp/README.md)
- [ ] (선택) `mes/api/server.py` FastAPI 추론 서비스 분리 — 현재 빈 스텁
- 제외(결정): 결함 유형(Type1~5) **자동분류** — 위치 기반 분류가 무작위 수준(≈20%). [decisions.md](decisions.md) §8
