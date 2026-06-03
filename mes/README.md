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

## 미확정 결정사항 (구현 전 확정 필요)

1. **유형(Type1~5) 표시** — 모델은 good/bad·위치는 주지만 유형 분류는 직접 안 함.
   ㉠ 히트맵 위치 기반 간단 규칙 분류 추가 / ㉡ 유형 없이 good/bad+위치만.
2. **대시보드 형태** — Streamlit 단일앱(채택 예정, 가장 빠름) vs FastAPI+웹.

> 현재는 **뼈대만** 존재. 각 파일의 TODO를 단계별로 채운다.
