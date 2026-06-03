"""④ 라인 시뮬레이터 — test 이미지를 컨베이어처럼 1장씩 추론·기록.

실제 카메라 대신 data/screw/test 의 사진을 config.FEED_INTERVAL_SEC 간격으로 흘려보낸다.
각 장: predict_one(추론+히트맵) → repository.insert(DB 기록). 대시보드는 이 DB를 읽는다.

유입 비율(good_ratio): test셋은 평가용이라 결함이 75%(119/160)지만, 실제 라인은 양품이
대부분이다. 그래서 test셋의 같은 사진을 쓰되 '양품을 더 자주 뽑도록' 비율만 현실화한다
(데이터 왜곡이 아니라 유입 빈도 조정 — 복원추출). good_ratio=None이면 test 분포 그대로.

실행:
  python -m mes.simulator.line_feeder                 # 80장, 1초 간격, 양품 85%
  python -m mes.simulator.line_feeder 120 0.5 0.9      # 120장, 0.5초 간격, 양품 90%
  python -m mes.simulator.line_feeder inf 1.0 0.85     # 무한, 양품 85%
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
import glob
import random
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from mes import config
from mes.core.inference import ScrewInferencer
from mes.storage import repository

DEFAULT_COUNT = 80
DEFAULT_GOOD_RATIO = 0.85


def _split_images():
    """test 이미지를 (양품, 불량) 리스트로 분리."""
    goods, bads = [], []
    for p in sorted(glob.glob(str(config.FEED_SOURCE / "*" / "*.png"))):
        (goods if Path(p).parent.name == "good" else bads).append(p)
    return goods, bads


def run(count=DEFAULT_COUNT, interval=None, good_ratio=DEFAULT_GOOD_RATIO, reset=True):
    interval = config.FEED_INTERVAL_SEC if interval is None else interval
    goods, bads = _split_images()
    if not goods or not bads:
        print(f"[오류] 이미지 부족: good={len(goods)} bad={len(bads)} ({config.FEED_SOURCE})")
        return

    infinite = (count == "inf")
    total = None if infinite else int(count)

    def pick():
        if good_ratio is None:                      # test 분포 그대로(복원추출)
            return random.choice(goods + bads)
        return random.choice(goods if random.random() < good_ratio else bads)

    inf = ScrewInferencer()
    repository.reset() if reset else repository.init_db()

    ratio_txt = "분포충실" if good_ratio is None else f"양품 {good_ratio:.0%}"
    print(f"[라인 시작] 소스={config.FEED_SOURCE.name}  간격={interval}s  "
          f"{'무한' if infinite else f'{total}장'}  유입={ratio_txt}  (Ctrl+C 중지)")

    n_good = n_bad = i = 0
    try:
        while infinite or i < total:
            i += 1
            p = pick()
            r = inf.predict_one(p, render=True)
            repository.insert(r)
            if r["verdict"] == "bad":
                n_bad += 1
            else:
                n_good += 1
            flag = "⛔ 불량" if r["verdict"] == "bad" else "✅ 양품"
            tail = f"[{i}/{total}]" if total else f"[{i}]"
            print(f"  {tail} {flag}  score={r['score']:.3f}  "
                  f"누적불량률={n_bad/i:5.1%}  ({Path(p).parent.name}/{Path(p).name})", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[중지] 사용자 중단")

    print(f"\n[라인 종료] 처리 {i}장  양품 {n_good} / 불량 {n_bad}  불량률 {n_bad/max(i,1):.1%}")
    print(f"  DB: {config.DB_PATH}  (대시보드: streamlit run mes/dashboard/app.py)")


def _parse_argv():
    count = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COUNT
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else None
    good_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_GOOD_RATIO
    if count != "inf":
        count = int(count)
    return count, interval, good_ratio


if __name__ == "__main__":
    c, iv, gr = _parse_argv()
    run(count=c, interval=iv, good_ratio=gr)
