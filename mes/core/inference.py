"""① 추론 코어 — 학습된 PatchCore(.pt)로 사진 1장을 판정한다.

대시보드/시뮬레이터/API 가 공통으로 쓰는 가장 안쪽 모듈. 서버가 뜰 때 모델을 1회 로드해
메모리에 상주시킨다(매번 학습 X). export_model.py 로 만든 배포용 .pt 가 필요하다.

  predict_one(image) -> {
      image_id, ts, verdict('good'|'bad'), score(이상점수), heatmap_path
  }

* 결함 유형(Type1~5) 자동분류는 제외했다: 이 데이터에서 PatchCore의 anomaly map은 위치는
  잘 짚지만(pixel AUROC 0.98) 가장 뜨거운 점이 유형과 무관하게 중앙으로 쏠려, 위치 기반
  유형 분류가 무작위 수준(≈20%)으로 불가함을 데이터로 확인했다(유형별 축위치 t가 0.46~0.55로 완전 겹침).
  대신 히트맵이 결함 '위치'를 시각적으로 보여준다.

셀프테스트: python -m mes.core.inference
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRUST_REMOTE_CODE", "1")   # TorchInferencer(.pt) pickle 로드 허용
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from mes import config

_RES = config.IMAGE_SIZE


def render_heatmap(rgb_320, amap_320, out_path):
    """원본(320 RGB) 위에 anomaly_map을 jet 컬러맵으로 오버레이해 PNG 저장.

    이상도에 비례한 per-pixel alpha 로 합성 → 배경(저이상도)은 원본에 가깝게,
    뜨거운 영역만 색이 진해져 결함 위치가 깔끔하게 부각된다.
    """
    from matplotlib import cm
    a = np.asarray(amap_320, dtype=np.float64)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    heat = cm.jet(a)[..., :3] * 255.0                          # (H,W,3) float
    base = np.asarray(rgb_320, dtype=np.float64)
    alpha = (0.75 * a)[..., None]                              # 이상도 비례 불투명도
    blend = ((1 - alpha) * base + alpha * heat).astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blend).save(out_path)
    return str(out_path)


class ScrewInferencer:
    """배포용 .pt 를 로드해 1장씩 추론하는 상주 객체."""

    def __init__(self, pt_path=None, device="auto"):
        pt_path = Path(pt_path or config.EXPORTED_PT)
        if not pt_path.exists():
            raise FileNotFoundError(
                f"배포용 모델(.pt)이 없습니다: {pt_path}\n"
                f"  -> 먼저 `python -m mes.core.export_model` 로 생성하세요."
            )
        from anomalib.deploy import TorchInferencer
        self.inferencer = TorchInferencer(str(pt_path), device=device)
        config.HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    def predict_one(self, image, image_id=None, render=True):
        """이미지(경로 또는 PIL) 1장 -> 판정 결과 dict."""
        if image_id is None:
            if isinstance(image, (str, Path)):
                p = Path(image)
                image_id = f"{p.parent.name}_{p.stem}"   # 폴더_파일명 (충돌 방지)
            else:
                image_id = datetime.now().strftime("%H%M%S_%f")

        batch = self.inferencer.predict(image)
        score = float(batch.pred_score)
        verdict = "bad" if score >= config.ANOMALY_THRESHOLD else "good"
        amap = batch.anomaly_map.squeeze().detach().cpu().numpy()   # (320,320)

        heatmap_path = None
        if render:
            # 렌더용 원본 320 RGB
            pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) \
                else image.convert("RGB")
            rgb = np.asarray(pil.resize((_RES, _RES)))
            heatmap_path = render_heatmap(rgb, amap, config.HEATMAP_DIR / f"{image_id}.png")

        return {
            "image_id": image_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "verdict": verdict,
            "score": round(score, 4),
            "heatmap_path": heatmap_path,
        }


# --------------------------------------------------------------------------
# 셀프테스트 / 유형규칙 보정
# --------------------------------------------------------------------------
def _selftest():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import glob

    inf = ScrewInferencer()
    test_dir = config.FEED_SOURCE
    print(f"[셀프테스트] {test_dir}  임계값={config.ANOMALY_THRESHOLD}")

    # 1) 샘플 몇 장 상세 출력 (히트맵 생성)
    print("\n-- 샘플 추론 (히트맵 생성) --")
    for folder in ["good", "thread_side", "scratch_head", "manipulated_front"]:
        fs = sorted(glob.glob(str(test_dir / folder / "*.png")))
        if fs:
            r = inf.predict_one(fs[0], render=True)
            print(f"  {folder:18s} -> verdict={r['verdict']:4s} score={r['score']:.3f}  "
                  f"heatmap={Path(r['heatmap_path']).name}")

    # 2) 전체 검출 혼동행렬
    print("\n-- 전체 검출 집계 --")
    tp = fp = fn = tn = 0
    for folder in sorted(os.listdir(test_dir)):
        for p in sorted(glob.glob(str(test_dir / folder / "*.png"))):
            is_bad = inf.predict_one(p, render=False)["verdict"] == "bad"
            if folder == "good":
                fp += is_bad; tn += (not is_bad)
            else:
                tp += is_bad; fn += (not is_bad)
    rec = tp / (tp + fn) if tp + fn else 0
    prec = tp / (tp + fp) if tp + fp else 0
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}  Recall={rec:.3f} Precision={prec:.3f}")


if __name__ == "__main__":
    _selftest()
