"""나사 정렬 crop — 나사를 PCA 주축으로 세워(머리가 위) 고정 크기 캔버스에 담는다.

왜: baseline 은 1024px 이미지를 256→224 로 줄여 작은 결함(약 40px)이 10px 안팎이 된다. 나사는 이미지의 일부(길이 약 900px, 폭 최대 약 283px)라서
나사만 세워 잘라내면 같은 계산량에서 해상도를 2배 이상 올릴 수 있고, 회전 변동도 사라진다.

- 전경은 synth 의 정제 전경(배경보다 90 이상 어두운 픽셀)을 쓰되, 반사로 나사가 조각나는 경우(480장 중 11장)를 막기 위해
  큰 close 로 조각을 잇고 큰 조각들을 모두 유지한다(`robust_fg`). synth.py 는 이미 만든 합성 풀의 재현성 때문에 건드리지 않는다.
- 캔버스(원본 px 기준 320×960) → 배율 scale 로 축소해 모델에 입력한다. 머리 방향은 synth.screw_axis 의 "폭이 넓은 쪽"을 따른다.
- CAM 은 정렬 프레임에서 나오므로 `cam_to_eval_frame` 으로 원본 프레임에 역투영한 뒤 baseline 과 같은 224 중앙 crop 으로 맞춰
  GT 마스크(loc_eval.load_mask)와 비교한다 → baseline 지표와 같은 기준.
"""
import functools

import numpy as np
import cv2

from src.supervised import synth as S

CANVAS_W, CANVAS_H = 320, 960      # 원본 px 기준 캔버스(나사 길이 ≤ 약 913, 머리 폭 ≤ 약 283)


def robust_fg(g, drop=90, close=41, keep=0.05):
    fo = S.foreground_mask(g)
    bg = np.median(g[fo == 0])
    c = ((cv2.GaussianBlur(g, (0, 0), 2) < bg - drop) & (fo > 0)).astype(np.uint8) * 255
    c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(c)
    if n <= 1:
        return fo
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep_ids = [k + 1 for k in range(n - 1) if st[k + 1, cv2.CC_STAT_AREA] >= keep * big]
    m = np.isin(lab, keep_ids).astype(np.uint8) * 255
    cn, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f = np.zeros_like(m)
    cv2.drawContours(f, cn, -1, 255, -1)
    return f


@functools.lru_cache(maxsize=2048)
def align_full(path):
    """이미지 → (정렬 캔버스 gray [CANVAS_H, CANVAS_W], 원본→캔버스 아핀 M(2x3), 나사 길이, 원본 크기)."""
    g = S.read_gray(path)
    f = robust_fg(g)
    ax = S.screw_axis(f)
    bg = int(np.median(g[f == 0]))
    pt = ax['c'] + (ax['tmin'] + ax['tmax']) / 2 * ax['v']          # 나사 축 중간점
    M = cv2.getRotationMatrix2D((float(pt[0]), float(pt[1])), ax['theta'] - 90.0, 1.0)   # 머리→끝 방향이 아래(+y)가 되도록
    M[0, 2] += CANVAS_W / 2 - pt[0]
    M[1, 2] += CANVAS_H / 2 - pt[1]
    canvas = cv2.warpAffine(g, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    return canvas, M, float(ax['tmax'] - ax['tmin']), g.shape


def align_scaled(path, scale):
    """정렬 캔버스를 scale 배로 축소한 RGB uint8 [H, W, 3] (H=CANVAS_H*scale, W=CANVAS_W*scale)."""
    canvas = align_full(path)[0]
    w, h = round(CANVAS_W * scale), round(CANVAS_H * scale)
    small = canvas if scale == 1 else cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    return np.stack([small] * 3, -1)


def cam_to_eval_frame(cam, path, out=224):
    """정렬 프레임의 CAM [h, w] → 원본 프레임 → baseline 평가 프레임(Resize256 + CenterCrop224)의 [224, 224]."""
    _, M, _, shape = align_full(path)
    c = cv2.resize(cam.astype(np.float32), (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_LINEAR)
    full = cv2.warpAffine(c, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    small = cv2.resize(full, (256, 256), interpolation=cv2.INTER_AREA)
    o = (256 - out) // 2
    return small[o:o + out, o:o + out]
