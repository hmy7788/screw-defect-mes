"""⑤ 관제 대시보드 — Streamlit.

storage(SQLite)를 주기적으로 읽어 실시간 생산현황을 보여준다.
시뮬레이터(mes.simulator.line_feeder)가 채우는 DB를 그대로 시각화한다.

실행:
  1) (한 터미널) python -m mes.simulator.line_feeder inf 1.0 0.9
  2) (다른 터미널) streamlit run mes/dashboard/app.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
from pathlib import Path

# `streamlit run mes/dashboard/app.py` 는 스크립트 폴더만 sys.path에 넣어 `import mes`가 실패한다.
# 레포 루트(이 파일 기준 3단계 위)를 경로에 추가해 어떤 실행 방식이든 동작하게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from mes import config
from mes.storage import repository

st.set_page_config(page_title="나사 결함 검사 관제", page_icon="🔩", layout="wide")
repository.init_db()

# ---- 사이드바: 제어 ----
st.sidebar.title("🔩 관제 설정")
auto = st.sidebar.checkbox("자동 새로고침", value=True)
refresh = st.sidebar.slider("새로고침 간격(초)", 1, 10, 2)
window = st.sidebar.slider("이동 불량률 구간", 5, 50, 20)
st.sidebar.caption("불량 판정 임계값(이상점수) = "
                   f"{config.ANOMALY_THRESHOLD} · 모델 PatchCore(coreset 0.01/320)")
if st.sidebar.button("DB 초기화(데모 재시작)"):
    repository.reset()
    st.sidebar.success("초기화됨")

# ---- 데이터 로드 ----
stats = repository.stats()
hist = repository.history()
bad_recent = repository.recent(limit=8, only_bad=True)

st.title("나사 결함 검사 — 실시간 관제")
st.caption(f"갱신 {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  입력원 = data/screw/test (라인 시뮬레이션)")

if stats["total"] == 0:
    st.info("아직 검사 데이터가 없습니다. 다른 터미널에서 시뮬레이터를 실행하세요:\n\n"
            "`python -m mes.simulator.line_feeder inf 1.0 0.9`")
    if auto:
        time.sleep(refresh)
        st.rerun()
    st.stop()

# ---- 최신 검사 알람 ----
latest = hist[-1]
if latest["verdict"] == "bad":
    st.error(f"⛔ 불량 감지 — {latest['image_id']}  (이상점수 {latest['score']:.3f}, {latest['ts']})")
else:
    st.success(f"✅ 정상 — {latest['image_id']}  (이상점수 {latest['score']:.3f}, {latest['ts']})")

# ---- KPI ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("누적 생산", f"{stats['total']:,}")
c2.metric("양품", f"{stats['good']:,}")
c3.metric("불량", f"{stats['bad']:,}")
c4.metric("불량률", f"{stats['defect_rate']:.1%}")

# ---- 추이 차트 ----
df = pd.DataFrame(hist)
df["순번"] = range(1, len(df) + 1)
df["불량"] = (df["verdict"] == "bad").astype(int)
df["누적 불량률"] = df["불량"].cumsum() / df["순번"]
df["이동 불량률"] = df["불량"].rolling(window, min_periods=1).mean()
df["이상점수"] = df["score"]

left, right = st.columns(2)
with left:
    st.subheader("불량률 추이")
    st.line_chart(df.set_index("순번")[["누적 불량률", "이동 불량률"]])
with right:
    st.subheader("이상점수 추이")
    chart = df.set_index("순번")[["이상점수"]].copy()
    chart["임계값"] = config.ANOMALY_THRESHOLD
    st.line_chart(chart)

# ---- 최근 불량 히트맵 갤러리 ----
st.subheader("최근 불량 — 결함 위치 히트맵")
if bad_recent:
    cols = st.columns(4)
    for i, r in enumerate(bad_recent):
        with cols[i % 4]:
            hp = r.get("heatmap_path")
            if hp and Path(hp).exists():
                st.image(hp, use_container_width=True)
            st.caption(f"{r['image_id']}  ·  {r['score']:.3f}")
else:
    st.write("아직 불량이 없습니다.")

# ---- 최근 검사 목록 ----
with st.expander("최근 검사 목록(20)"):
    recent = repository.recent(limit=20)
    rdf = pd.DataFrame(recent)[["ts", "image_id", "verdict", "score"]]
    st.dataframe(rdf, use_container_width=True, hide_index=True)

# ---- 자동 새로고침 ----
if auto:
    time.sleep(refresh)
    st.rerun()
