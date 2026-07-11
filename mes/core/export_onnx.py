"""학습된 PatchCore 체크포인트를 ONNX 로 export (C# MES용, 1회성).

C# MES(ScrewMes)는 외부 Python 프로세스 없이 ONNX Runtime으로 직접 추론한다.

[중요 발견 — _onnx_probe.py 진단 결과]
anomalib 2.5의 ExportType.ONNX 는 전처리(Resize+Normalize)·후처리(점수 정규화+임계)를
**그래프에 모두 내장**한다. 출력이 pred_score/pred_label/anomaly_map/pred_mask 로 나오고,
입력에 '원본 RGB를 255로 나눈 [0,1] CHW 텐서'(리사이즈·정규화 안 함)를 그대로 넣으면
onnx 의 pred_score 가 Python predict_one 의 score 와 정확히 일치한다(오차 0.0000).
 → C#은 별도 정규화/리사이즈 없이: 원본 이미지 → /255 → CHW → onnx → pred_score 에
   임계값 0.513 적용 → good/bad. anomaly_map 으로 히트맵.

이 스크립트는 model.onnx 를 만들고, 그 사실을 meta.json 에 적고,
parity_samples.json(검증 정답: onnx 점수 == python 점수)을 남긴다.

선행: `python -m mes.core.export_model`  (predict_one 정답 산출에 .pt 필요)
실행: `python -m mes.core.export_onnx`
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRUST_REMOTE_CODE", "1")
import sys
import json
import glob
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from PIL import Image

from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.deploy import ExportType

from mes import config
from mes.core.export_model import make_pp, MEAN, STD
from mes.core.inference import ScrewInferencer

RES = config.IMAGE_SIZE
OUT_DIR = config.ROOT / "csharp" / "model"        # C# 솔루션이 읽는 산출물 위치
N_PARITY = 36                                      # 파리티 표본 수 (good/결함 섞어서)


def export_onnx():
    """ckpt → model.onnx. 반환: onnx 파일 경로."""
    model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                      coreset_sampling_ratio=0.01, pre_processor=make_pp(RES))
    engine = Engine(accelerator="gpu", devices=1)
    print(f"[export] ckpt={config.MODEL_CKPT}")
    print(f"[export] -> {OUT_DIR}")
    engine.export(model=model, export_type=ExportType.ONNX,
                  ckpt_path=str(config.MODEL_CKPT), export_root=str(OUT_DIR))
    hits = glob.glob(str(OUT_DIR / "**" / "*.onnx"), recursive=True)
    if not hits:
        print("[오류] model.onnx 를 찾지 못했습니다."); sys.exit(1)
    onnx_path = Path(sorted(hits, key=len)[0])
    print(f"[완료] onnx: {onnx_path}")
    return onnx_path


def to_input(path):
    """원본 RGB → /255 → CHW float (리사이즈·정규화는 onnx 그래프가 수행)."""
    x = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0    # HWC [0,1]
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None], np.float32)


def main():
    if not config.MODEL_CKPT.exists():
        print(f"[오류] ckpt 없음: {config.MODEL_CKPT}"); sys.exit(1)

    onnx_path = export_onnx()

    try:
        import onnxruntime as ort
    except ImportError:
        print("[오류] onnxruntime 미설치 → `pip install onnxruntime` 후 재실행"); sys.exit(1)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print(f"[onnx] input={in_name} {sess.get_inputs()[0].shape}  outputs={out_names}")
    score_out = "pred_score" if "pred_score" in out_names else out_names[0]
    map_out = "anomaly_map" if "anomaly_map" in out_names else out_names[-1]

    # 파리티 표본: good 절반 + 결함 절반
    test = config.FEED_SOURCE
    goods = sorted(glob.glob(str(test / "good" / "*.png")))
    bads = [p for p in sorted(glob.glob(str(test / "*" / "*.png")))
            if Path(p).parent.name != "good"]
    rng = np.random.default_rng(0)
    half = N_PARITY // 2
    sample = list(rng.choice(goods, min(half, len(goods)), replace=False)) + \
             list(rng.choice(bads, min(N_PARITY - half, len(bads)), replace=False))

    inf = ScrewInferencer()                       # 정답 = production 경로(predict_one)
    samples, errs, agree = [], [], 0
    for p in sample:
        py = inf.predict_one(p, render=False)
        out = sess.run([score_out], {in_name: to_input(p)})
        onnx_score = float(np.asarray(out[0]).reshape(-1)[0])
        onnx_verdict = "bad" if onnx_score >= config.ANOMALY_THRESHOLD else "good"
        errs.append(abs(onnx_score - py["score"]))
        agree += int(onnx_verdict == py["verdict"])
        samples.append({"image_id": f"{Path(p).parent.name}_{Path(p).stem}",
                        "onnx_score": onnx_score, "py_score": py["score"],
                        "verdict": py["verdict"]})
    max_err = float(np.max(errs))
    print(f"\n[파리티] 표본 {len(sample)}장  최대|onnx-py|={max_err:.5f}  "
          f"평균={np.mean(errs):.5f}  판정일치 {agree}/{len(sample)}")

    try:
        import anomalib; ver = anomalib.__version__
    except Exception:
        ver = "unknown"

    meta = {
        "image_size": RES,
        "preprocess": {                            # onnx 그래프가 리사이즈+정규화 내장
            "scale": "div255", "layout": "NCHW",
            "resize_in_graph": True, "normalize_in_graph": True,
            "mean": MEAN, "std": STD,              # 참고용(C#은 적용 불필요)
        },
        "onnx": {"input": in_name, "outputs": out_names,
                 "score_output": score_out, "map_output": map_out},
        "threshold": config.ANOMALY_THRESHOLD,     # 정규화점수(pred_score)에 직접 적용
        "max_abs_err": max_err,
        "anomalib_version": ver,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "parity_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"[기록] {OUT_DIR/'meta.json'}")
    print(f"[기록] {OUT_DIR/'parity_samples.json'}  ({len(samples)}건)")

    if max_err > 0.01 or agree < len(sample):
        print("\n⚠️  오차/불일치 → ONNX 네이티브 경로 재검토(REST 폴백 고려).")
    else:
        print("\n✅  파리티 양호 → C# ONNX 네이티브 진행(go).")


if __name__ == "__main__":
    main()
