"""③ 결과 DB 헬퍼 — SQLite 기록/조회.

PoC 라 SQLite 파일 1개(config.DB_PATH)로 충분.
"""
# TODO(구현):
#   init_db():            schema.sql 실행해 테이블 생성
#   insert(record: dict): 검사 1건 기록
#   recent(limit):        최근 N건 조회 (대시보드용)
#   stats():              누적 생산수/불량수/유형분포 집계


def init_db():
    raise NotImplementedError  # TODO


def insert(record):
    raise NotImplementedError  # TODO
