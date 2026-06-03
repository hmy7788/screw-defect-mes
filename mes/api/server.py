"""② 추론 서비스 — FastAPI.

이미지를 받아 core.inference.predict_one 으로 판정하고 결과를 JSON 으로 반환한다.
서버 기동 시 모델을 1회 로드해 상주시킨다.

실행: uvicorn mes.api.server:app --port 8000
"""
# TODO(구현):
#   startup 이벤트에서 core.load_model() 1회 호출
#   POST /predict  (multipart 이미지) -> predict_one 결과 + storage 기록
#   GET  /health   -> 모델 로드 상태

# from fastapi import FastAPI
# app = FastAPI(title="Screw Defect MES - Inference")
