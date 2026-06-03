# MES-lite PoC — 나사 결함 검사 자동화 + 관제

연구 트랙(`../src`)에서 학습한 **PatchCore 이상탐지 모델**을 실제 생산 라인처럼 운용하는
데모(PoC). 진짜 공장 MES의 거대한 범위 중 **품질검사 자동화 + 관제** 부분만 떼어낸 경량 버전이다.

## 데이터 흐름

```
④ simulator        ② api(추론)         ③ storage        ⑤ dashboard
test 이미지를  ──→  PatchCore로    ──→  판정 결과를  ──→  실시간 화면:
1장씩 전송         good/bad+점수        SQLite에         생산현황·불량률 추이
(라인 시뮬)        +히트맵+유형         한 줄씩 기록      ·유형분포·불량 알람
                      ↑
                  ① core/inference (모델 1장 추론)
```

## 구성

| 폴더 | 역할 |
|------|------|
| `core/` | 학습된 모델 로드 + 사진 1장 추론(점수·히트맵·판정·유형) |
| `api/` | 추론을 HTTP 서비스로 노출 (FastAPI, `POST /predict`) |
| `storage/` | 판정 결과 DB (SQLite) |
| `simulator/` | test 이미지를 시간 간격 두고 1장씩 흘려보내는 라인 시뮬레이터 |
| `dashboard/` | 관제 대시보드 (Streamlit) |
| `runtime/` | 런타임 산출물(db 파일·히트맵 png) — gitignore |

## 실행 순서 (구현 완료 후)

```bash
# 0) (최초 1회) 모델 준비 — 기존 학습 ckpt 재사용 or 새로 학습
# 1) 추론 API 띄우기
uvicorn mes.api.server:app --port 8000
# 2) 대시보드 띄우기
streamlit run mes/dashboard/app.py
# 3) 라인 시뮬레이터 시작 (이미지 흘려보내기)
python -m mes.simulator.line_feeder
```

## 진행 상황

- ✅ **Phase 1 추론 코어 완료** (`core/export_model.py`, `core/inference.py`)
  - ckpt(coreset 0.01/320) → 배포용 `.pt` export → `ScrewInferencer.predict_one(이미지)`
    → `{verdict, score, heatmap_path}`
  - good/bad 판정 임계값 0.513 (F2 최적, FN 최소화): **Recall 0.992, Precision 0.929 (FP 9/41)**
  - 히트맵: 이상도 비례 alpha 오버레이로 결함 위치 부각, `runtime/heatmaps/<id>.png` 저장
- ⬜ Phase 2 storage / 3 simulator / 4 dashboard / 5 api(선택)

## 결정 사항

1. **유형(Type1~5) 자동분류 — 제외.** 데이터로 검증한 결과, PatchCore anomaly map은 위치는
   잘 짚지만(pixel AUROC 0.98) 가장 뜨거운 점이 유형과 무관하게 중앙으로 쏠려, 위치 기반 유형
   분류가 무작위 수준(≈20%, 유형별 축위치 t가 0.46~0.55로 완전 겹침)이라 불가. → good/bad + 위치
   히트맵만 제공(히트맵이 결함 위치를 시각적으로 보여줌).
2. **대시보드 형태** — Streamlit 단일앱(채택 예정, 가장 빠름).
3. **모델 ckpt** — coreset 0.01 / 해상도 320 (빠른 추론 + 위치 AUPRO 0.956).

## 실행 (Phase 1)

```bash
python -m mes.core.export_model   # 최초 1회: ckpt -> .pt
python -m mes.core.inference      # 셀프테스트 (검출 집계 + 샘플 히트맵)
```
