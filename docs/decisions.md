# 프로젝트 결정·문제 기록

> 이 프로젝트를 진행하면서 마주친 문제, 선택의 기로, 그리고 그 근거를 정리한 문서.
> 면접에서 "왜 이렇게 했어요?" 질문에 답하기 위한 자기 언어로 된 기록.
> 전체 그림은 [overview.md](overview.md), 트랙별 상세는 [baseline.md](baseline.md) · [anomaly.md](anomaly.md).

---

## 1. 왜 이진분류 → 이상탐지로 바꿨나?

**문제**: 기존 졸업논문은 good/bad 이진분류(ResNet-18). 결함이 유형당 train 20장뿐이라
학습 데이터가 너무 적었고, Type1(나사끝)·Type4(나사산 측면) 같은 미세 결함을 잘 못 잡았다.

**선택**: 비지도 이상탐지(PatchCore)로 전환.
- **정상(good) 이미지만 학습** → "결함 유형당 20장" 문제가 학습 단계에서 사라짐
- 결함 119장 **전부를 평가**에 쓸 수 있음 (기존엔 학습에 묶여 있었음)
- MVTec 마스크로 **결함 위치를 픽셀 단위 정량 평가** 가능

**핵심 통찰**: 이 데이터셋이 MVTec AD screw와 동일하다는 걸 발견 → 공개 마스크를 그대로 사용.

---

## 2. 환경 설정에서 만난 함정들

### OMP 충돌 (`libiomp5md.dll`)
- **증상**: 실행 시 `OMP: Error #15` 에러
- **원인**: Windows에서 OpenMP 라이브러리가 중복 로드됨
- **해결**: 코드 상단에 `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` 추가

### anomalib Split enum 버그 (Python 3.11)
- **증상**: 학습 데이터가 0건으로 로드됨
- **원인**: anomalib의 `Split` enum을 pandas 필터에 쓰면 Python 3.11에서 0건 매칭
- **해결**: `src/anomaly/ad_data.py`의 `MVTecADFixed`에서 split을 문자열 `"train"/"test"`로 전달
- **교훈**: 라이브러리 버전 호환성 문제는 실제로 자주 생긴다. 데이터가 0건이면 로딩 문제부터 의심.

### AUPRO 계산 오류
- **증상**: anomalib의 `AUPRO`에 원시 텐서를 넣으면 실패
- **원인**: anomalib 2.5.0의 `AUPRO`가 Batch 필드 방식이라 원시 텐서를 못 받음
- **해결**: 내부 베이스 클래스 `_AUPRO`를 직접 사용 (`src/anomaly/ad_eval.py`)

---

## 3. 해상도 선택 — 256 vs 320 vs 512

**질문**: 해상도를 높이면 위치 정확도가 올라갈까?

**실험 결과**:
| 해상도 | AUPRO | Type4 위치 | 시간 |
|--------|-------|-----------|------|
| 256 | 0.941 | 0.971 | 146초 |
| 320 | **0.966** | **0.984** | 70분 |
| 512 | OOM | — | — |

**선택**: 320이 최고 정확도지만 70분은 비현실적. → coreset 0.01로 9분으로 단축해 현실화.

**512 OOM 이유**: coreset 크기 문제가 아니라 추론 시 거리계산 텐서가 7.5GB를 통째로 요구. 
당시 사용한 8GB GPU(RTX 2070 SUPER)에서 원천적으로 불가. → "512는 이 GPU에서 비현실적"으로 확정하고 접음.

**핵심 결론**: 해상도↑ → Type4(나사산 측면) 위치가 가장 크게 개선됨. 미세·국소 결함에 고해상도가 유효함을 정량 확인.

---

## 4. coreset 0.1 vs 0.01

**coreset이 뭔가**: PatchCore의 메모리뱅크에 저장할 대표 패치 비율.
높을수록 정밀하지만 느리고 메모리를 많이 씀.

**선택의 이유**:
- coreset 0.1: 정확도 최고(AUPRO 0.966)지만 320에서 70분
- coreset 0.01: AUPRO 0.956(−0.01), 320에서 **9분(7.8배↓)**

**결론**: 손실이 미미하고 시간이 7.8배 줄어드니 0.01이 실용적. 단 "최고 정확도"는 0.1.
MES 추론에는 0.01/320 ckpt 채택(빠른 실시간 추론).

---

## 5. 모델 비교 — PatchCore vs EfficientAD vs PaDiM

**왜 비교했나**: PatchCore만 쓰면 "왜 이 모델이에요?"에 답하기 어려움.
다른 모델과 비교해야 선택의 근거가 생김.

**결과**:
| 모델 | Image AUROC | AUPRO | 특이사항 |
|------|-------------|-------|---------|
| **PatchCore** | **0.96** | **0.966** | 주력 |
| EfficientAD | 0.85~0.93 | 0.935 | 학습형, batch=1 강제, imagenette 1.5GB 다운로드 필요 |
| PaDiM | 0.76 | 0.908 | Recall 1.0이나 정상 오경보 40/41장 |

**EfficientAD에서 만난 함정**:
- 전처리에 Normalize 넣으면 에러 (`"Transforms should not contain Normalize"`)
- 이유: EfficientAD가 정규화를 내부에서 처리함
- 해결: `make_pp(res, normalize=False)` — Resize만 적용

**결론**: PatchCore가 검출·위치 양면 최선.

---

## 6. MES 추론 — 어떻게 단일 이미지를 추론하나

**문제**: anomalib은 DataModule을 통해 배치로 추론하는 게 기본. 이미지 1장을 바로 넣는 방법이 필요.

**선택**: `engine.export(ExportType.TORCH)` → `model.pt` 생성 후 `TorchInferencer` 사용.
- `.pt`에 전처리(Resize+Normalize)·후처리(정규화+임계값)가 **모두 내장**됨
- 원본 이미지를 그대로 넣어도 동작 → 실용적

**주의**: `TRUST_REMOTE_CODE=1` 환경변수 필요 (pickle 로드 보안 경고 때문).

---

## 7. good/bad 임계값 — 왜 0.513인가

**문제**: `.pt` 내장 임계값은 정상 오경보가 너무 많음 (FP 16/41).

**과정**:
1. test셋 전체(good 41 + 결함 119)에 추론 실행
2. F2 점수를 최대화하는 임계값 탐색 → **0.513**
3. 결과: TP=118 FP=9 FN=1, Recall 0.992

**왜 F2인가**: F2는 Recall(불량을 잡는 비율)을 Precision(오경보 적음)보다 4배 더 중시.
→ "불량을 놓치느니 정상을 재검사한다"는 제조 현장의 철학과 일치.

**trade-off**: 임계값을 올리면 오경보↓ 하지만 불량 놓침↑. 0.513은 FN 최소화 기조.

---

## 8. 결함 유형(Type1~5) 자동분류 — 왜 포기했나

**시도**: 히트맵의 가장 뜨거운 점(결함 위치)을 나사 주축에 투영해 유형을 추정.
- 나사를 전경 분할 → PCA로 주축 추출 → 결함 위치를 축에 투영 → t(머리=0~끝=1), r(반경)으로 분류

**결과**: **유형 일치율 20% (5지선다 무작위와 동일)**

**원인 분석**: 유형별 t 분포를 찍어보니 전부 0.46~0.55에 몰려있었음.
Type1(끝)도, Type2(머리)도 모두 중간값. PatchCore 히트맵이 결함 위치는 잘 짚지만
가장 뜨거운 점이 유형과 무관하게 중앙으로 쏠리는 경향이 있기 때문.

**결론**: 튜닝으로 해결 불가 → 유형 분류 제외. good/bad + 위치 히트맵만 제공.
→ **데이터로 확인해서 빼는 결정**이 코드 많이 짜는 것보다 낫다.

---

## 9. 시뮬레이터 불량률이 너무 높았던 이유

**문제**: 시뮬레이터를 돌리니 불량률이 87%로 비현실적.

**원인**: MVTec test셋은 **평가용**이라 결함이 75%(119/160). 실제 공장은 양품이 대부분.

**해결**: 같은 test셋을 쓰되 양품을 더 자주 뽑도록 **유입 비율(good_ratio)만 조정**.
데이터를 바꾸는 게 아니라 "복원추출 빈도"만 현실화. 기본값 85%.

---

## 10. 대시보드에서 만난 버그들

### import 경로 오류
- **증상**: `streamlit run mes/dashboard/app.py` 시 `No module named 'mes'`
- **원인**: `streamlit run`은 스크립트 폴더만 `sys.path`에 추가해 레포 루트를 못 찾음
- **해결**: `app.py` 상단에 `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`

### 빈 DB에서 KeyError
- **증상**: DB가 비었을 때 `pd.DataFrame([])[["ts", ...]]`에서 KeyError
- **원인**: 빈 리스트로 만든 DataFrame엔 컬럼이 없음
- **해결**: `if recent:` 조건 추가

---

## 11. baseline 을 레포에 편입하고 데이터 중복을 없앤 이유

**문제**: baseline(졸업논문)이 별도 레포 `screw_defect`에 있어 "baseline → 한계 → 고도화" 서사가 한 곳에서 안 보였다.
편입 과정에서 데이터가 `data/screw`(MVTec 원본) · `data/new_k-fold_data` · `data/new_k-fold_data_gt` 세 벌(약 566MB)로 늘었다.

**발견**: `new_k-fold_data` 480장은 `data/screw` 와 **바이트 단위로 100% 동일**(해시 480/480 일치)한 폴더 재배치 복사본이고,
`_gt` 는 거기에 MVTec 마스크를 얹은 파생물이었다(마스크 119개도 원본과 동일).

**선택**: 이미지는 `data/screw` 한 벌만 두고, baseline 분할을 **매니페스트 한 파일**(`splits/baseline_split.csv`, 480행:
split·class·filename·image·mask)로 표현. 로더는 `src/supervised/data.py` 로 단일화했다.
- 약 377MB 중복 제거, git 에는 24KB CSV 만 추적(데이터 `data/` 는 계속 gitignore).
- 과거 `'good' in path`, `'type1' in path` 같은 **경로 문자열 파싱**으로 유형을 판별하던 코드를 `class_of(path)`(매니페스트 조회)로 교체 —
  폴더명에 의존하지 않아 안전해졌다.

**검증(리팩터링 회귀)**: 정렬 순서 = 과거 `sorted(os.listdir)` 와 동일하도록 filename 순으로 정렬. 구 레이아웃의 출력을 기준값으로 저장해 비교했다 —
train/test 목록·라벨, 반복 홀드아웃 분할(seed 42), 5-Fold 인덱스, train/val 분할, 이미지 480장·마스크 119개 바이트, `class_of` 전수 일치.
Grad-CAM 위치평가도 소수점 4자리까지 동일(Pixel AUROC 0.9213 / AUPRO 0.8142).

**교훈**: "복사본 폴더"는 파생물일 수 있다 — 해시로 원본과의 관계를 먼저 확인하면 저장소를 크게 줄이고 분할 정의를 코드/데이터로 명시할 수 있다.

---

## 12. 디렉터리를 트랙별로 정리한 기준

- **`src/` 는 목적별 패키지**(`common` / `supervised` / `anomaly`): baseline(분류기 학습)과 이상탐지(anomalib 래퍼)는 함께 import 되는 일이 없어 분리했다.
  과거에는 루트 `src/` 와 `baseline/src/` 가 같은 이름의 패키지라 합치면 충돌할 상황이었다(`utils.py` 는 동일 내용이라 `common` 으로 단일화).
- **실행 스크립트는 `experiments/`**: 라이브러리(`src`)와 분리. 스크립트는 `__file__` 기준으로 레포 루트를 `sys.path` 에 넣고 모든 경로를 `src/common/paths.py` 에서 가져와
  **실행 위치와 무관**하게 동작한다(과거 `root="data"`, `"outputs/..."` 같은 cwd 상대경로 제거).
- **결과 구분**: 커밋하는 결과는 `results/supervised|anomaly/`, 원본 산출물은 `outputs/`(gitignore). `mes/config.py` 의 ckpt 경로(`outputs/sweep_cs001/...`)는 스윕 스크립트 출력 규칙과 결합돼 있어 유지했다.
- **문서는 `docs/` 로 통합**: 흩어져 있던 PLAN(낡은 계획)·HANDOFF·HANDOVER·README 를 최신 상태 기준으로 `overview / baseline / anomaly / decisions` 로 재구성.
  컴포넌트 README(`mes/`, `csharp/`)는 코드 옆에 둔다.
- **미사용 코드 정리**: `src/mvtec_data.py`(MVTec 직접 재분할 시도 후 중단), 루트 `src/utils.py`(중복), 구 `build_kfold_ground_truth.py`(매니페스트로 대체)를 제거했다.

---

## 13. README 자산을 만들며 알게 된 것

- **재학습하면 수치가 바뀐다.** README 용 모델을 다시 학습(coreset 0.01/320)했더니 Image AUROC 0.970 / AUPRO 0.954 / Recall 1.0(스윕 기준)으로 원본(0.964 / 0.956 / 0.992)과 같은 범위였지만,
  임계값 0.513 에서 정상 오경보가 9장 → 15장으로 늘었다(Recall 은 0.992 로 동일). README 에는 재현 가능한 수치를 쓰고 편차를 명시했다. 임계값을 모델마다 다시 맞추면 test 로 튜닝·평가하는 문제가 더 커지므로 0.513 을 유지했다.
- **baseline 도 재학습하면 흔들린다.** 논문 Test F2 0.9896 인 ResNet-18 을 같은 설정으로 재학습하면 0.904, 저장된 `best_model.pth` 는 0.885. test 가 55장(결함 19장)이라 분산이 크다 — MobileNet repeated holdout(0.818 ± 0.044)과 같은 이야기다.
- **비교 그림은 선별하지 않는다.** "baseline 은 틀리고 PatchCore 는 맞는" 장면만 고르면 편향이다. 유형별 test 첫 장(고정 규칙)을 썼고, 그 결과 PatchCore 의 정상 오경보(325번 점수 0.58)와 배경 오반응도 그대로 보인다.
- **anomalib 은 비ASCII 절대경로를 거부한다.** 한글 사용자 폴더에서 `data/screw` 절대경로를 넘기면 `validate_path` 가 실패 → 상대경로로 변환(`build_screw_datamodule`).
- **의존성 누락**: ONNX export 에 `onnx` 가 필요한데 requirements 에 없었다 → 추가. `build_patchcore_summary.py` 는 pandas 버전에 따라 CSV 공백 때문에 merge 가 실패해 공백을 제거하도록 고쳤다.

---

## 면접 핵심 답변 요약

**"왜 이상탐지인가?"**
→ 결함 데이터가 부족하고(유형당 20장), 미세 결함(Type4)이 안 잡혔다.
이상탐지는 정상만 학습하므로 데이터 부족 문제가 사라지고, 결함을 전부 평가에 쓸 수 있다.

**"왜 PatchCore인가?"**
→ EfficientAD·PaDiM과 비교 실험했고, Image AUROC 0.96·AUPRO 0.966으로 양면 최선이었다.

**"임계값을 왜 0.513으로 했나?"**
→ test셋에서 F2를 최대화하는 값을 탐색했다. F2는 불량 미검출(FN)을 오경보(FP)보다 중시하는 지표로,
제조 현장에서 불량 출하가 재검사보다 훨씬 치명적이기 때문에 이 방침을 계승했다.

**"아쉬운 점은?"**
→ 결함 유형 자동분류를 시도했지만 데이터 분석 결과 무작위 수준(20%)으로 불가능했다.
결함 히트맵이 결함 위치는 잘 짚지만, 유형 구분에 필요한 세밀한 위치 차이를 잡지 못했다.

**"실제 공장에 배포하려면?"**
→ 현재는 test 데이터 시뮬레이션이지만, 실제 적용엔 카메라 이미지 수집 파이프라인,
모델 재학습(공장 데이터로 fine-tuning), 더 엄격한 임계값 검증이 필요하다.

**"기존 지도학습 baseline 은 어떻게 처리했나?"**
→ 버리지 않고 레포에 편입해 비교 기준선으로 남겼다. baseline 은 분류는 잘하지만(Test F2 0.9896) Grad-CAM 위치평가에서 Type4 가 Pixel AUROC 0.65 / AUPRO 0.45 로 무너졌고,
이상탐지(PatchCore)는 같은 유형을 0.98 수준으로 짚는다(단 split·표본 수가 달라 엄밀한 직접 비교는 아님을 명시).
