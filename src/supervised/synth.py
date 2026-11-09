"""Cut & Paste 합성 결함 생성 — MVTec screw 의 GT 마스크로 결함 영역만 잘라 정상 나사의 같은 해부학적 위치에 이식한다.

핵심 아이디어
- 나사는 이미지마다 회전돼 있으므로 위치를 나사 자체 좌표 (t, s) 로 표현한다.
  전경 분할 → PCA 주축 → t(0=머리, 1=끝, 머리는 폭이 넓은 쪽), s(폭 방향 -1~1).
- 소스 결함의 (t, s) 를 대상 정상 나사의 같은 (t, s) 로 옮기고, 두 나사의 축 각도 차이만큼 patch 를 회전 + 지터/스케일.
- cv2.seamlessClone(Poisson blending) 으로 경계 이음선을 없앤다.
- 전경은 Otsu 로 얻은 뒤 "배경보다 확실히 어두운 픽셀"로 한 번 더 정제한다(Otsu 전경은 그림자까지 포함해 좌표가 틀어짐).

소스는 호출자가 넘긴 레코드만 쓴다 — test 결함은 절대 넘기지 말 것(평가 누수).
cv2.imread 는 한글 경로를 못 읽으므로 fromfile+imdecode / imencode+tofile 을 쓴다.
"""
import os
import csv
import math
import functools
from dataclasses import dataclass

import numpy as np
import cv2

from src.common.paths import REPO_ROOT

TYPES = ['type1', 'type2', 'type3', 'type4', 'type5']
TYPE_FOLDER = {'type1': 'manipulated_front', 'type2': 'scratch_head', 'type3': 'scratch_neck',
               'type4': 'thread_side', 'type5': 'thread_top'}


@dataclass
class SynthConfig:
    t_tol: float = 0.04               # 소스 결함 t 위치 허용 오차(0~1 정규화)
    s_tol: float = 0.15               # 폭 방향 s 허용 오차
    rot_jitter: float = 10.0          # 회전 지터(도)
    scale_range: tuple = (0.85, 1.15)
    pad: int = 14                     # 결함 bbox 주변 여유(px) — Poisson 이 주변 맥락을 쓰도록
    dilate: int = 5                   # 클론 영역 = 결함 마스크를 이만큼 팽창
    clone_mode: str = 'normal'        # 'normal' | 'mixed' | 'naive'(단순 덮어쓰기, 비교용)
    min_fg_frac: float = 0.9          # 합성 마스크가 "나사 몸체 근방"에 있어야 하는 최소 비율
    near_px: int = 31                 # 몸체 근방 = 전경 팽창 폭(manipulated_front 는 끝이 결손돼 마스크가 몸체 밖으로 나감)
    core_drop: int = 90               # 배경보다 이만큼 어두워야 "나사 몸체"(그림자 제외)
    largest_only: bool = True         # GT 마스크가 여러 조각이면 가장 큰 조각만 이식


# ---------------------------------------------------------------- 입출력
def read_gray(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img is not None, path
    return img


def write_png(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode('.png', img)
    assert ok
    buf.tofile(path)


# ---------------------------------------------------------------- 전경 / 축 좌표계
def foreground_mask(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() / 255 > 0.5:            # 전경이 소수 면적이 되도록 극성 정렬
        bw = 255 - bw
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        return np.zeros_like(bw)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return filled


def tight_foreground(g, fg, core_drop):
    bg = np.median(g[fg == 0])
    c = ((cv2.GaussianBlur(g, (0, 0), 2) < bg - core_drop) & (fg > 0)).astype(np.uint8) * 255
    c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(c)
    if n <= 1:
        return fg
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    c = (lab == k).astype(np.uint8) * 255
    cn, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f = np.zeros_like(c)
    cv2.drawContours(f, cn, -1, 255, -1)   # 밝은 반사로 생긴 내부 구멍 메움
    return f


NBINS = 40
TC = (np.arange(NBINS) + 0.5) / NBINS


def _profile(P, c, v):
    n = np.array([-v[1], v[0]], dtype=np.float32)
    a = (P - c) @ v
    b = (P - c) @ n
    tmin, tmax = float(a.min()), float(a.max())
    idx = np.clip(((a - tmin) / (tmax - tmin) * NBINS).astype(int), 0, NBINS - 1)
    lo = np.full(NBINS, np.nan)
    hi = np.full(NBINS, np.nan)
    for k in range(NBINS):
        sel = idx == k
        if sel.any():
            lo[k], hi[k] = b[sel].min(), b[sel].max()
    ok = ~np.isnan(lo)
    lo = np.interp(TC, TC[ok], lo[ok])
    hi = np.interp(TC, TC[ok], hi[ok])
    return n, tmin, tmax, lo, hi


def screw_axis(fg):
    ys, xs = np.nonzero(fg)
    P = np.stack([xs, ys], 1).astype(np.float32)
    c = P.mean(0)
    _, V = np.linalg.eigh(np.cov((P - c).T))
    v = V[:, 1].astype(np.float32)
    n, tmin, tmax, lo, hi = _profile(P, c, v)
    w = hi - lo
    q = NBINS // 4
    if w[-q:].mean() > w[:q].mean():     # 머리(폭이 넓은 쪽)가 t=0 이 되도록 뒤집기
        v = -v
        n, tmin, tmax, lo, hi = _profile(P, c, v)
    return dict(c=c, v=v, n=n, tmin=tmin, tmax=tmax, mid=(lo + hi) / 2, width=hi - lo,
                theta=math.degrees(math.atan2(v[1], v[0])))


def to_ts(ax, pt):
    d = np.asarray(pt, np.float32) - ax['c']
    t = (float(d @ ax['v']) - ax['tmin']) / (ax['tmax'] - ax['tmin'])
    mid = np.interp(t, TC, ax['mid'])
    w = np.interp(t, TC, ax['width'])
    return t, (float(d @ ax['n']) - mid) / (w / 2 + 1e-6)


def from_ts(ax, t, s):
    mid = np.interp(t, TC, ax['mid'])
    w = np.interp(t, TC, ax['width'])
    a = ax['tmin'] + t * (ax['tmax'] - ax['tmin'])
    return ax['c'] + a * ax['v'] + (mid + s * w / 2) * ax['n']


@functools.lru_cache(maxsize=1024)
def analyze(path, core_drop=90):
    """이미지 → (gray, 나사 몸체 전경 마스크, 축 좌표계). 결과는 캐시된다."""
    g = read_gray(path)
    fg = tight_foreground(g, foreground_mask(g), core_drop)
    return g, fg, screw_axis(fg)


# ---------------------------------------------------------------- patch 추출 / 변환 / 이식
def largest_component(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 2:
        return m
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def extract_patch(gray, mask, cfg):
    m = mask > 127
    if cfg.largest_only:
        m = largest_component(m)
    ys, xs = np.nonzero(m)
    x0, x1 = max(xs.min() - cfg.pad, 0), min(xs.max() + cfg.pad + 1, gray.shape[1])
    y0, y1 = max(ys.min() - cfg.pad, 0), min(ys.max() + cfg.pad + 1, gray.shape[0])
    mm = (m[y0:y1, x0:x1].astype(np.uint8)) * 255
    clone = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.dilate + 1,) * 2))
    return dict(img=gray[y0:y1, x0:x1].copy(), mask=mm, clone=clone,
                centroid=np.array([xs.mean() - x0, ys.mean() - y0], np.float32))


def transform_patch(p, angle, scale):
    h, w = p['img'].shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cs, sn = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sn + w * cs) + 2, int(h * cs + w * sn) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    img = cv2.warpAffine(p['img'], M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(p['mask'], M, (nw, nh), flags=cv2.INTER_LINEAR) > 127
    clone = cv2.warpAffine(p['clone'], M, (nw, nh), flags=cv2.INTER_LINEAR) > 127
    cen = M @ np.array([*p['centroid'], 1.0])
    # 클론 영역 bbox 로 자르되 크기를 홀수로 맞춰 seamlessClone 중심을 정확히 지정
    ys, xs = np.nonzero(clone)
    x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, nw)
    y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, nh)
    if (x1 - x0) % 2 == 0:
        x1 = x1 + 1 if x1 < nw else x1 - 1
    if (y1 - y0) % 2 == 0:
        y1 = y1 + 1 if y1 < nh else y1 - 1
    sl = (slice(y0, y1), slice(x0, x1))
    return dict(img=img[sl], mask=(mask[sl] * 255).astype(np.uint8), clone=(clone[sl] * 255).astype(np.uint8),
                centroid=(cen - [x0, y0]).astype(np.float32))


def place(shape, tp, target_pt):
    """patch 를 target_pt 에 놓을 때의 (x0, y0). 이미지 경계를 벗어나면 None. (seamlessClone 전에 싸게 검사하기 위해 분리)"""
    h, w = tp['img'].shape
    H, W = shape
    center = np.round(np.asarray(target_pt) - (tp['centroid'] - [w // 2, h // 2])).astype(int)
    x0, y0 = center[0] - w // 2, center[1] - h // 2
    if x0 < 1 or y0 < 1 or x0 + w > W - 1 or y0 + h > H - 1:
        return None
    return int(x0), int(y0), center


def gt_mask(shape, tp, x0, y0):
    h, w = tp['img'].shape
    gt = np.zeros(shape, np.uint8)
    gt[y0:y0 + h, x0:x0 + w] = tp['mask']
    return gt


def paste(dst_gray, tp, x0, y0, center, mode):
    """결함 patch 를 dst 에 합성해 합성 이미지를 반환한다."""
    h, w = tp['img'].shape
    if mode == 'naive':
        out = dst_gray.copy()
        roi = out[y0:y0 + h, x0:x0 + w]
        roi[tp['clone'] > 0] = tp['img'][tp['clone'] > 0]
        return out
    flag = cv2.NORMAL_CLONE if mode == 'normal' else cv2.MIXED_CLONE
    src3 = cv2.cvtColor(tp['img'], cv2.COLOR_GRAY2BGR)
    dst3 = cv2.cvtColor(dst_gray, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(cv2.seamlessClone(src3, dst3, tp['clone'], (int(center[0]), int(center[1])), flag),
                        cv2.COLOR_BGR2GRAY)


def generate_one(src_rec, good_rec, seed, cfg=None, mode=None, tries=30):
    """소스 결함 레코드(path/mask) 1건을 정상 레코드(path) 1건에 이식. 실패 시 None."""
    cfg = cfg or SynthConfig()
    mode = mode or cfg.clone_mode
    rng = np.random.default_rng(seed)
    sg, _, sax = analyze(src_rec['path'], cfg.core_drop)
    mask = read_gray(src_rec['mask'])
    patch = extract_patch(sg, mask, cfg)
    mm = largest_component(mask > 127) if cfg.largest_only else (mask > 127)
    ys, xs = np.nonzero(mm)
    t0, s0 = to_ts(sax, (xs.mean(), ys.mean()))
    gg, gfg, gax = analyze(good_rec['path'], cfg.core_drop)
    near = cv2.dilate(gfg, np.ones((cfg.near_px, cfg.near_px), np.uint8))
    for _ in range(tries):
        t = float(np.clip(t0 + rng.uniform(-cfg.t_tol, cfg.t_tol), 0.02, 0.98))
        s = float(np.clip(s0 + rng.uniform(-cfg.s_tol, cfg.s_tol), -0.9, 0.9))
        angle = (sax['theta'] - gax['theta']) + rng.uniform(-cfg.rot_jitter, cfg.rot_jitter)
        scale = rng.uniform(*cfg.scale_range)
        pt = from_ts(gax, t, s)
        if gfg[int(round(pt[1])), int(round(pt[0]))] == 0:      # 이식 중심은 반드시 몸체 위
            continue
        tp = transform_patch(patch, angle, scale)
        pl = place(gg.shape, tp, pt)
        if pl is None:
            continue
        x0, y0, center = pl
        gt = gt_mask(gg.shape, tp, x0, y0)
        if (near[gt > 0] > 0).mean() < cfg.min_fg_frac:
            continue
        out = paste(gg, tp, x0, y0, center, mode)               # 느린 Poisson blending 은 검사 통과 후에만
        return dict(img=out, mask=gt, src=src_rec, good=good_rec, t=t, s=s, t_src=t0, s_src=s0,
                    angle=angle, scale=scale, patch=patch, tpatch=tp, pt=pt)
    return None


# ---------------------------------------------------------------- 풀 생성 / 로드 / 배분
CSV_FIELDS = ['synth_file', 'synth_mask', 'class', 'source_image', 'source_mask', 'target_good',
              't', 's', 'angle', 'scale', 'seed', 'mode']


def _rel(p):
    return os.path.relpath(p, REPO_ROOT).replace(os.sep, '/')


def build_pool(sources_by_type, goods, per_type, cfg, seed, out_dir, csv_path):
    """유형별로 per_type 장의 합성 결함을 만들어 out_dir/<type>/ 에 저장하고 csv_path 에 정의를 기록한다.
    소스는 유형 내 round-robin, 대상 정상은 랜덤. 앞에서부터 n 장을 쓰면 비율이 다른 실험들이 서로 포함 관계(nested)가 된다.
    반환: {type: [record(path, mask, type, label=1, ...), ...]}"""
    pool = {t: [] for t in sources_by_type}
    rows = []
    for ti, t in enumerate(sources_by_type):
        srcs = sources_by_type[t]
        attempts, slot_fail = 0, 0
        fails, oks = {}, {}
        while len(pool[t]) < per_type:
            attempts += 1
            if attempts > per_type * 30:
                raise RuntimeError(f'{t}: 합성 생성이 계속 실패 — 설정(t_tol/near_px 등)을 확인하세요')
            src = srcs[(len(pool[t]) + slot_fail) % len(srcs)]   # 실패하면 다음 소스로
            rng = np.random.default_rng(seed * 100003 + ti * 1009 + attempts)
            good = goods[int(rng.integers(len(goods)))]
            sd = int(rng.integers(1 << 30))
            r = generate_one(src, good, sd, cfg)
            if r is None:
                fails[src['path']] = fails.get(src['path'], 0) + 1
                slot_fail += 1 if fails[src['path']] % 3 == 0 else 0   # 같은 소스가 3번 연속 실패하면 다음 소스로
                continue
            slot_fail = 0
            oks[src['path']] = oks.get(src['path'], 0) + 1
            idx = len(pool[t])
            img_p = os.path.join(out_dir, t, f'{t}_{idx:03d}.png')
            msk_p = os.path.join(out_dir, t, f'{t}_{idx:03d}_mask.png')
            write_png(img_p, r['img'])
            write_png(msk_p, r['mask'])
            pool[t].append(dict(path=img_p, mask=msk_p, type=t, label=1, source=src['path']))
            rows.append(dict(synth_file=_rel(img_p), synth_mask=_rel(msk_p), **{'class': t},
                             source_image=_rel(src['path']), source_mask=_rel(src['mask']),
                             target_good=_rel(good['path']), t=f"{r['t']:.4f}", s=f"{r['s']:.4f}",
                             angle=f"{r['angle']:.2f}", scale=f"{r['scale']:.3f}", seed=sd, mode=cfg.clone_mode))
        unusable = [os.path.basename(p) for p in fails if p not in oks]
        if unusable:
            print(f'[알림] {t}: 이식 불가 소스(결함이 나사 몸체 밖에 걸침) {unusable} — 건너뜀')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return pool


def load_pool(csv_path):
    """build_pool 이 남긴 CSV 에서 풀을 복원. 파일이 하나라도 없으면 None."""
    if not os.path.exists(csv_path):
        return None
    pool = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            p = os.path.join(REPO_ROOT, *r['synth_file'].split('/'))
            m = os.path.join(REPO_ROOT, *r['synth_mask'].split('/'))
            if not (os.path.exists(p) and os.path.exists(m)):
                return None
            pool.setdefault(r['class'], []).append(
                dict(path=p, mask=m, type=r['class'], label=1,
                     source=os.path.join(REPO_ROOT, *r['source_image'].split('/'))))
    return pool


def allocate(n_normal, ratio, types=TYPES):
    """합성 총량 = round(정상 수 × ratio) 를 유형에 균등 배분(나머지는 앞 유형부터). {type: n}"""
    total = round(n_normal * ratio)
    base, rem = divmod(total, len(types))
    return {t: base + (1 if i < rem else 0) for i, t in enumerate(types)}


def take(pool, alloc):
    """풀에서 유형별로 앞 alloc[t] 장씩 → 레코드 리스트."""
    return [r for t in alloc for r in pool[t][:alloc[t]]]
