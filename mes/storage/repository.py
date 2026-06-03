"""③ 결과 DB 헬퍼 — SQLite 기록/조회.

PoC 라 SQLite 파일 1개(config.DB_PATH)로 충분.
검사 1건 = inspections 테이블 한 행. 대시보드의 모든 지표는 여기서 나온다.

  init_db()         : schema.sql 실행(테이블 생성)
  insert(record)    : predict_one 결과 dict 1건 기록
  recent(n)         : 최근 n건 (대시보드 최신 목록/갤러리)
  history()         : 시간 오름차순 전체 (불량률 추이 차트)
  stats()           : 누적 생산수/불량수/불량률
  reset()           : 테이블 비우기(데모 재시작용)

셀프테스트: python -m mes.storage.repository
"""
import sqlite3
from pathlib import Path

from mes import config

_SCHEMA = Path(__file__).with_name("schema.sql")


def _connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """schema.sql 로 테이블/인덱스 생성(이미 있으면 무시)."""
    with _connect() as conn:
        conn.executescript(_SCHEMA.read_text(encoding="utf-8"))


def insert(record: dict) -> int:
    """검사 1건 기록. predict_one 결과 dict를 그대로 받는다. 새 row id 반환."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO inspections (ts, image_id, verdict, score, heatmap_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (record["ts"], record["image_id"], record["verdict"],
             float(record["score"]), record.get("heatmap_path")),
        )
        conn.commit()
        return cur.lastrowid


def recent(limit: int = 20, only_bad: bool = False) -> list[dict]:
    """최근 검사 limit건 (최신순). only_bad=True면 불량만."""
    q = "SELECT * FROM inspections"
    if only_bad:
        q += " WHERE verdict = 'bad'"
    q += " ORDER BY id DESC LIMIT ?"
    with _connect() as conn:
        return [dict(r) for r in conn.execute(q, (limit,)).fetchall()]


def history(limit: int | None = None) -> list[dict]:
    """시간 오름차순 전체(또는 마지막 limit건) — 불량률 추이 차트용."""
    if limit is None:
        q = "SELECT * FROM inspections ORDER BY id ASC"
        args = ()
    else:
        # 마지막 limit건을 시간순으로
        q = "SELECT * FROM (SELECT * FROM inspections ORDER BY id DESC LIMIT ?) ORDER BY id ASC"
        args = (limit,)
    with _connect() as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def stats() -> dict:
    """누적 생산수/불량수/양품수/불량률."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN verdict='bad' THEN 1 ELSE 0 END) AS bad FROM inspections"
        ).fetchone()
    total = row["total"] or 0
    bad = row["bad"] or 0
    return {
        "total": total,
        "bad": bad,
        "good": total - bad,
        "defect_rate": (bad / total) if total else 0.0,
    }


def reset():
    """테이블 비우기(데모 재시작용)."""
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM inspections")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='inspections'")
        conn.commit()


def _selftest():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from datetime import datetime
    reset()
    print(f"[셀프테스트] DB={config.DB_PATH}")
    samples = [
        {"image_id": "good_000", "ts": datetime.now().isoformat(timespec="seconds"),
         "verdict": "good", "score": 0.42, "heatmap_path": None},
        {"image_id": "thread_top_003", "ts": datetime.now().isoformat(timespec="seconds"),
         "verdict": "bad", "score": 0.88, "heatmap_path": "runtime/heatmaps/thread_top_003.png"},
        {"image_id": "good_011", "ts": datetime.now().isoformat(timespec="seconds"),
         "verdict": "good", "score": 0.47, "heatmap_path": None},
    ]
    for r in samples:
        rid = insert(r)
        print(f"  insert id={rid}: {r['image_id']} ({r['verdict']})")
    print("  stats:", stats())
    print("  recent(2):", [r["image_id"] for r in recent(2)])
    print("  recent only_bad:", [r["image_id"] for r in recent(only_bad=True)])
    print("  history count:", len(history()))
    assert stats()["total"] == 3 and stats()["bad"] == 1
    print("  OK ✅ (재실행 시 reset되어 항상 3건)")


if __name__ == "__main__":
    _selftest()
