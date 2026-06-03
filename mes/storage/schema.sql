-- ③ 결과 DB 스키마 (SQLite)
-- 검사 1건 = 한 행. 대시보드의 모든 그래프는 이 테이블에서 나온다.
-- (결함 유형 분류는 데이터상 불가로 제외 — good/bad + 위치 히트맵만 저장)

CREATE TABLE IF NOT EXISTS inspections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,   -- 검사 시각 (ISO8601)
    image_id    TEXT    NOT NULL,   -- 원본 이미지 식별자 (폴더_파일명)
    verdict     TEXT    NOT NULL,   -- 'good' | 'bad'
    score       REAL    NOT NULL,   -- 이상점수 (정규화)
    heatmap_path TEXT               -- 히트맵 png 경로
);

CREATE INDEX IF NOT EXISTS idx_inspections_ts ON inspections(ts);
