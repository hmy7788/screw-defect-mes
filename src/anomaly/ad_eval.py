"""이상탐지 평가·분석 레이어.

anomalib predict 결과로부터:
  - 이미지 단위: AUROC, F2(임계값 튜닝), Recall, Precision  (FN 최소화 기조 계승)
  - 픽셀 단위: Pixel AUROC, AUPRO  (결함 '위치' 정량 평가, gt_mask 사용)
  - 유형별(Type1~5): 검출률(recall) + 위치 품질(mean per-image pixel AUROC)
    → Type1/Type4 개선을 숫자로 확인하는 핵심 표
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, fbeta_score, recall_score, precision_score, confusion_matrix,
)

# MVTec screw 결함 폴더 -> 논문 유형명
DEFECT_TO_TYPE = {
    "good": "good",
    "manipulated_front": "Type1",  # 나사끝 휘어짐/끊어짐
    "scratch_head": "Type2",       # 나사머리 깨짐
    "scratch_neck": "Type3",       # 나사목 깨짐
    "thread_side": "Type4",        # 나사산 측면
    "thread_top": "Type5",         # 나사산 상단
}
TYPE_ORDER = ["Type1", "Type2", "Type3", "Type4", "Type5"]


def collect_predictions(engine, model, dataloader):
    """engine.predict 결과를 numpy 배열로 수집."""
    preds = engine.predict(model=model, dataloaders=dataloader)
    scores, labels, types, amaps, gmasks = [], [], [], [], []
    for b in preds:
        scores.append(b.pred_score.detach().cpu().numpy().reshape(-1))
        labels.append(b.gt_label.detach().cpu().numpy().astype(int).reshape(-1))
        amaps.append(b.anomaly_map.detach().cpu().numpy())          # (B,H,W)
        gmasks.append(b.gt_mask.detach().cpu().numpy().astype(int))  # (B,H,W)
        for p in b.image_path:
            folder = os.path.basename(os.path.dirname(p))
            types.append(DEFECT_TO_TYPE.get(folder, folder))
    return dict(
        scores=np.concatenate(scores),
        labels=np.concatenate(labels),
        types=np.array(types),
        amaps=np.concatenate(amaps, axis=0),
        gmasks=np.concatenate(gmasks, axis=0),
    )


def tune_threshold_f2(scores, labels, beta=2):
    """F2 최대화 임계값 탐색 (불량 미검출 최소화 기조)."""
    cands = np.unique(scores)
    best_thr, best_f2 = float(cands[0]), -1.0
    for t in cands:
        pred = (scores >= t).astype(int)
        f2 = fbeta_score(labels, pred, beta=beta, pos_label=1, zero_division=0)
        if f2 > best_f2:
            best_f2, best_thr = f2, float(t)
    return best_thr, best_f2


def image_metrics(scores, labels, thr):
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return dict(
        image_AUROC=float(roc_auc_score(labels, scores)),
        F2=float(fbeta_score(labels, pred, beta=2, pos_label=1, zero_division=0)),
        Recall=float(recall_score(labels, pred, pos_label=1, zero_division=0)),
        Precision=float(precision_score(labels, pred, pos_label=1, zero_division=0)),
        threshold=float(thr), TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
    )


def _per_image_pixel_auroc(amap, gmask):
    """한 장의 픽셀 AUROC (마스크에 양/음 픽셀이 모두 있을 때만)."""
    y = gmask.reshape(-1)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, amap.reshape(-1))


def pixel_metrics(amaps, gmasks):
    """전체 테스트셋 픽셀 단위 위치 평가."""
    y = gmasks.reshape(-1)
    s = amaps.reshape(-1)
    out = {"pixel_AUROC": float(roc_auc_score(y, s)) if y.min() != y.max() else np.nan}
    # AUPRO — anomalib의 순수 torchmetrics 베이스(_AUPRO) 사용(원시 텐서)
    try:
        import torch
        from anomalib.metrics.aupro import _AUPRO
        m = _AUPRO()
        m.update(torch.from_numpy(amaps).float(), torch.from_numpy(gmasks).int())
        out["AUPRO"] = float(m.compute())
    except Exception as e:
        out["AUPRO"] = float("nan")
        out["AUPRO_error"] = str(e)[:120]
    return out


def per_type_table(data, thr):
    """유형별 검출률 + 위치 품질 표."""
    scores, types = data["scores"], data["types"]
    amaps, gmasks = data["amaps"], data["gmasks"]
    rows = []
    for t in TYPE_ORDER:
        idx = np.where(types == t)[0]
        if len(idx) == 0:
            continue
        det = float(np.mean(scores[idx] >= thr))  # 모두 불량이므로 검출률=recall
        loc = [_per_image_pixel_auroc(amaps[i], gmasks[i]) for i in idx]
        loc = np.array(loc, dtype=float)
        rows.append({
            "Type": t, "n": len(idx),
            "검출률": round(det, 4),
            "탐지수": f"{int((scores[idx] >= thr).sum())}/{len(idx)}",
            "위치_pixelAUROC": round(float(np.nanmean(loc)), 4),
            "평균점수": round(float(np.mean(scores[idx])), 4),
        })
    return pd.DataFrame(rows)


def evaluate_all(data, beta=2):
    """전체 평가 묶음. (summary dict, per_type DataFrame) 반환."""
    thr, _ = tune_threshold_f2(data["scores"], data["labels"], beta=beta)
    summary = image_metrics(data["scores"], data["labels"], thr)
    summary.update(pixel_metrics(data["amaps"], data["gmasks"]))
    table = per_type_table(data, thr)
    return summary, table
