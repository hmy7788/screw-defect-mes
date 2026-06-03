"""⑤ 관제 대시보드 — Streamlit.

storage(SQLite)를 주기적으로 읽어 실시간 생산현황을 보여준다.

실행: streamlit run mes/dashboard/app.py
"""
# TODO(구현) — 화면 구성:
#   상단 KPI:   누적 생산수 / 불량수 / 불량률
#   추이 그래프: 시간별 불량률 꺾은선
#   유형 분포:  Type1~5 막대 (결정사항 ① 채택 시)
#   불량 갤러리: 최근 불량 이미지 + 히트맵
#   알람:       신규 불량 발생 시 강조
#   (st.autorefresh 등으로 주기 갱신)

# import streamlit as st
# st.set_page_config(page_title="나사 결함 검사 관제", layout="wide")
