"""④ 라인 시뮬레이터 — test 이미지를 컨베이어처럼 1장씩 추론 API로 전송.

실제 카메라 대신 data/screw/test 의 사진을 config.FEED_INTERVAL_SEC 간격으로 흘려보낸다.
정상/불량을 섞어 전송해 "라인이 돌아가는" 상황을 재현한다.

실행: python -m mes.simulator.line_feeder
"""
# TODO(구현):
#   FEED_SOURCE 의 이미지를 순회(셔플 옵션)
#   각 이미지를 POST /predict 로 전송 (또는 core.predict_one 직접 호출)
#   FEED_INTERVAL_SEC 만큼 대기 반복


def run():
    raise NotImplementedError  # TODO


if __name__ == "__main__":
    run()
