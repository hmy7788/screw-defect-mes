"""① 추론 코어 — 학습된 PatchCore로 사진 1장을 판정한다.

대시보드/시뮬레이터/API 가 공통으로 쓰는 가장 안쪽 함수.
모델은 서버가 뜰 때 1회 로드하고 메모리에 상주시킨다(매번 학습 X).
"""
# TODO(구현):
#   load_model(): config.MODEL_CKPT 에서 PatchCore 로드 (anomalib)
#   predict_one(image) -> dict:
#       {
#         "verdict": "good" | "bad",       # ANOMALY_THRESHOLD 로 판정
#         "score":   float,                # 이상점수
#         "heatmap_path": str,             # 저장한 히트맵 png 경로
#         "type": str | None,              # 유형 추정 (결정사항 ①에 따라)
#       }


def load_model():
    """학습된 PatchCore 체크포인트를 로드해 반환 (1회 호출, 캐싱)."""
    raise NotImplementedError  # TODO


def predict_one(image):
    """이미지 1장 -> 판정 결과 dict."""
    raise NotImplementedError  # TODO
