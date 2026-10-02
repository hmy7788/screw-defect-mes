# MES-lite PoC — 나사 결함 검사 자동화 + 관제

연구 트랙(`../src`)에서 학습한 **PatchCore 이상탐지 모델**을 실제 생산 라인처럼 운용하는
데모(PoC). 진짜 공장 MES의 거대한 범위 중 **품질검사 자동화 + 관제** 부분만 떼어낸 경량 버전이다.

## 데이터 흐름

```
④ simulator        ① core(추론)        ③ storage        ⑤ dashboard
test 이미지를  ──→  PatchCore로    ──→  판정 결과를  ──→  실시간 화면:
1장씩 흘림         good/bad+점수        SQLite에         생산현황·불량률/점수 추이
(라인 시뮬)        +위치 히트맵         한 줄씩 기록      ·히트맵 갤러리·불량 알람

(② api/ — FastAPI 로 추론을 분리하는 선택 과제, 현재 빈 스텁)
```

## 구성

| 폴더 | 역할 |
|------|------|
| `core/` | 학습된 모델 로드 + 사진 1장 추론(점수·히트맵·good/bad 판정) + ONNX export |
| `api/` | (미구현 스텁) 추론을 HTTP 서비스로 노출 (FastAPI, `POST /predict`) |
| `storage/` | 판정 결과 DB (SQLite) |
| `simulator/` | test 이미지를 시간 간격 두고 1장씩 흘려보내는 라인 시뮬레이터 |
| `dashboard/` | 관제 대시보드 (Streamlit) |
| `runtime/` | 런타임 산출물(db 파일·히트맵 png) — gitignore |

## 실행 순서

```bash
# 0) (최초 1회) 학습 ckpt -> 배포용 .pt   (ckpt 없으면 python experiments/anomaly/run_resolution_sweep.py 0.01 로 생성, 약 9분)
python -m mes.core.export_model

# 1) (터미널 A) 라인 시뮬레이터: test 이미지를 1초당 1장, 양품 90% 비율로 흘려보냄
python -m mes.simulator.line_feeder inf 1.0 0.9

# 2) (터미널 B) 관제 대시보드
streamlit run mes/dashboard/app.py        # http://localhost:8501
```

## 진행 상황

- ✅ **Phase 1 추론 코어** (`core/export_model.py`, `core/inference.py`)
  - ckpt(coreset 0.01/320) → 배포용 `.pt` → `ScrewInferencer.predict_one` → `{verdict, score, heatmap_path}`
  - 임계값 0.513 (F2 최적, FN 최소화): **Recall 0.992, Precision 0.929 (FP 9/41)**
  - 히트맵: 이상도 비례 alpha 오버레이로 결함 위치 부각
- ✅ **Phase 2 storage** (`storage/repository.py`): SQLite init/insert/recent/history/stats/reset
- ✅ **Phase 3 simulator** (`simulator/line_feeder.py`): test 이미지 1장씩 추론·기록, 양품 비율 조정
- ✅ **Phase 4 dashboard** (`dashboard/app.py`): KPI·불량률 추이·이상점수 추이·불량 히트맵 갤러리·실시간 알람·자동 새로고침
- ⬜ Phase 5 api(선택): 추론을 FastAPI 서비스로 분리

## 결정 사항

1. **유형(Type1~5) 자동분류 — 제외.** PatchCore anomaly map은 위치는 잘 짚지만(pixel AUROC 0.98)
   가장 뜨거운 점이 유형과 무관하게 중앙으로 쏠려, 위치 기반 유형 분류가 무작위 수준(≈20%)이라 불가.
   → good/bad + 위치 히트맵만 제공.
2. **대시보드 형태** — Streamlit 단일앱.
3. **모델 ckpt** — coreset 0.01 / 해상도 320 (빠른 추론 + 위치 AUPRO 0.956).
4. **유입 비율** — test셋은 결함 75%(평가용)라, 시뮬레이터에서 양품 비율(good_ratio)을 현실화(기본 85%).
   단 모델 false-alarm(정상의 ~22%를 불량 판정, FN 최소화 임계값의 대가)으로 화면 불량률은 ~25%대로 보임.
