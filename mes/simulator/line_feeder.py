"""④ 라인 시뮬레이터 — test 이미지를 컨베이어처럼 1장씩 추론·기록.

실제 카메라 대신 data/screw/test 의 사진을 config.FEED_INTERVAL_SEC 간격으로 흘려보낸다.
각 장: predict_one(추론+히트맵) → repository.insert(DB 기록). 대시보드는 이 DB를 읽는다.
정상/불량이 섞여 흘러가 "라인이 돌아가는" 상황을 재현한다.

실행:
  python -m mes.simulator.line_feeder           # test 전체(160장) 1초 간격, DB 초기화 후
  python -m mes.simulator.line_feeder 40 0.5     # 40장, 0.5초 간격
  python -m mes.simulator.line_feeder inf 1.0     # 무한(이미지 순환), 1초 간격
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
import glob
import random
from itertools import cycle
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from mes import config
from mes.core.inference import ScrewInferencer
from mes.storage import repository


def _gather_images():
    fs = sorted(glob.glob(str(config.FEED_SOURCE / "*" / "*.png")))
    return fs


def run(count=None, interval=None, reset=True, shuffle=True):
    interval = config.FEED_INTERVAL_SEC if interval is None else interval
    images = _gather_images()
    if not images:
        print(f"[오류] 이미지 없음: {config.FEED_SOURCE}")
        return
    if shuffle:
        random.shuffle(images)

    infinite = count is None or (isinstance(count, str) and count == "inf")
    total = None if infinite else int(count)

    inf = ScrewInferencer()
    if reset:
        repository.reset()
    else:
        repository.init_db()

    print(f"[라인 시작] 소스={config.FEED_SOURCE.name}  간격={interval}s  "
          f"{'무한 순환' if infinite else f'{total}장'}  (Ctrl+C 중지)")

    src = cycle(images) if infinite else iter(images[:total] if total <= len(images)
                                              else [images[i % len(images)] for i in range(total)])
    n_good = n_bad = 0
    i = 0
    try:
        for p in src:
            i += 1
            r = inf.predict_one(p, render=True)
            repository.insert(r)
            if r["verdict"] == "bad":
                n_bad += 1
            else:
                n_good += 1
            rate = n_bad / i
            flag = "⛔ 불량" if r["verdict"] == "bad" else "✅ 양품"
            tail = f"  [{i}/{total}]" if total else f"  [{i}]"
            print(f"{tail} {flag}  score={r['score']:.3f}  "
                  f"불량률={rate:5.1%}  ({Path(p).parent.name}/{Path(p).name})", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[중지] 사용자 중단")

    print(f"\n[라인 종료] 처리 {i}장  양품 {n_good} / 불량 {n_bad}  불량률 {n_bad/max(i,1):.1%}")
    print(f"  DB: {config.DB_PATH}  (대시보드: streamlit run mes/dashboard/app.py)")


def _parse_argv():
    count = sys.argv[1] if len(sys.argv) > 1 else None       # 'inf' | 정수 | None(전체)
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else None
    if count is not None and count != "inf":
        count = int(count)
    return count, interval


if __name__ == "__main__":
    c, iv = _parse_argv()
    run(count=c, interval=iv)
