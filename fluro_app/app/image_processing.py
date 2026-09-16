# -*- coding: utf-8 -*-
"""图像处理模块：缩略图（M1）→ 预处理（M2）→ ROI 与特征（M3）逐步扩展。"""
import cv2
import numpy as np


def read_image(path):
    """读取图像为 RGB ndarray（BGR 转 RGB）。失败返回 None。"""
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_thumbnail(image, max_edge=320):
    """生成缩略图（保持比例，长边 max_edge）。image 为 RGB ndarray。"""
    h, w = image.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1.0:
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image.copy()


# ---------------- M2：预处理 ----------------

def _resize_ref(ref, target_shape):
    """参考图尺寸与原图不一致时缩放到一致（保持原始像素内容）。"""
    if ref is None:
        return None
    th, tw = target_shape[:2]
    if ref.shape[:2] != (th, tw):
        return cv2.resize(ref, (tw, th), interpolation=cv2.INTER_LINEAR)
    return ref


def dark_flat_correct(img_rgb, dark=None, flat=None):
    """暗场/平场校正：I' = (I - dark) / (flat - dark)。

    dark/flat 为 RGB ndarray 或 None；无参考图时跳过对应项。
    输出 0~255 uint8。
    """
    img = img_rgb.astype(np.float32)
    dark_f = _resize_ref(dark, img.shape).astype(np.float32) if dark is not None else None
    flat_f = _resize_ref(flat, img.shape).astype(np.float32) if flat is not None else None
    if dark_f is not None:
        img = img - dark_f
    if flat_f is not None:
        denom = flat_f - (dark_f if dark_f is not None else 0.0)
        denom = np.maximum(denom, 1e-6)
        img = img / denom * 255.0  # 还原到 0~255 亮度域
    return np.clip(img, 0, 255).astype(np.uint8)


def denoise(img_rgb, method='gaussian', kernel=5):
    """去噪：method ∈ gaussian|median；kernel 取奇数（自动向上取奇）。"""
    k = int(kernel)
    if k % 2 == 0:
        k += 1
    k = max(3, min(k, 31))
    if method == 'median':
        return cv2.medianBlur(img_rgb, k)
    return cv2.GaussianBlur(img_rgb, (k, k), 0)


def preprocess(img_rgb, method='gaussian', kernel=5, dark=None, flat=None):
    """组合预处理：暗场/平场校正 → 去噪。返回 RGB uint8。"""
    corrected = dark_flat_correct(img_rgb, dark=dark, flat=flat)
    return denoise(corrected, method=method, kernel=kernel)


# ---------------- M3：ROI 与特征 ----------------

def roi_to_pixels(roi, shape):
    """归一化 ROI (x,y,w,h) → 像素 (x0,y0,x1,y1)，越界裁剪。"""
    h, w = shape[:2]
    x, y, rw, rh = roi
    x0 = max(0, int(round(x * w)))
    y0 = max(0, int(round(y * h)))
    x1 = min(w, int(round((x + rw) * w)))
    y1 = min(h, int(round((y + rh) * h)))
    return x0, y0, x1, y1


def crop_roi(img_rgb, roi):
    """裁剪检测区；空区域抛 ValueError。"""
    x0, y0, x1, y1 = roi_to_pixels(roi, img_rgb.shape)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('ROI 区域为空或越界')
    return img_rgb[y0:y1, x0:x1]


def extract_features(img_rgb, roi):
    """提取检测区颜色与纹理特征。roi 为归一化 (x,y,w,h)。"""
    crop = crop_roi(img_rgb, roi)
    mean_bgr = crop.reshape(-1, 3).mean(axis=0)  # R,G,B
    mean_r, mean_g, mean_b = mean_bgr[0], mean_bgr[1], mean_bgr[2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    # 色相用圆形均值，避免红色端 0/180 环绕把均值拉偏
    hue_rad = np.deg2rad(hsv[:, :, 0].astype(np.float64) * 2.0)
    hue = float(np.rad2deg(np.arctan2(np.sin(hue_rad).mean(), np.cos(hue_rad).mean())) / 2.0)
    if hue < 0:
        hue += 180.0
    saturation = float(hsv[:, :, 1].mean())
    value = float(hsv[:, :, 2].mean())

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-12)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))

    return {
        'mean_r': round(float(mean_r), 3),
        'mean_g': round(float(mean_g), 3),
        'mean_b': round(float(mean_b), 3),
        'hue': round(hue, 3),
        'saturation': round(saturation, 3),
        'value': round(value, 3),
        'ratio_gr': round(float(mean_g) / (float(mean_r) + 1e-6), 4),
        'ratio_bg': round(float(mean_b) / (float(mean_g) + 1e-6), 4),
        'intensity': round(float(value), 3),
        'texture_entropy': round(entropy, 4),
    }


def auto_detect_roi(img_rgb, margin_frac=0.06):
    """基于内容自动识别检测区：取亮/饱和（荧光）像素的最大连通域外接矩形。

    返回归一化 (x,y,w,h)；找不到明显检测区返回 None。
    """
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0:
        return None
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    mask = ((s > 40) & (v > 50)).astype(np.uint8)
    k = max(3, min(31, (int(min(h, w) * 0.02) | 1)))
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    x0, y0, bw, bh = cv2.boundingRect(best)
    if bw * bh < h * w * 0.001:
        return None
    mx, my = int(margin_frac * bw), int(margin_frac * bh)
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w, x0 + bw + 2 * mx)
    y1 = min(h, y0 + bh + 2 * my)
    return (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)


def bg_ring_mean(img_rgb, roi, expand=0.25):
    """背景环均值：ROI 外扩 expand 比例的环形区域 RGB 均值（用于背景扣除）。"""
    h, w = img_rgb.shape[:2]
    x0, y0, x1, y1 = roi_to_pixels(roi, img_rgb.shape)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        raise ValueError('ROI 为空')
    ex0 = max(0, int(x0 - bw * expand))
    ey0 = max(0, int(y0 - bh * expand))
    ex1 = min(w, int(x1 + bw * expand))
    ey1 = min(h, int(y1 + bh * expand))
    if ex1 <= ex0 or ey1 <= ey0:
        raise ValueError('背景环为空')
    ox, oy = np.meshgrid(np.arange(ex0, ex1), np.arange(ey0, ey1))
    inner = (ox >= x0) & (ox < x1) & (oy >= y0) & (oy < y1)
    ring = img_rgb[ey0:ey1, ex0:ex1][~inner]
    if ring.size == 0:
        raise ValueError('背景环为空')
    return ring.reshape(-1, 3).mean(axis=0)


def apply_bg_subtraction(feats, bg_rgb):
    """背景扣除：从特征均值中减去背景环均值，并重算派生指标（色相/饱和度/明度/比值）。"""
    nr = max(0.0, float(feats['mean_r']) - float(bg_rgb[0]))
    ng = max(0.0, float(feats['mean_g']) - float(bg_rgb[1]))
    nb = max(0.0, float(feats['mean_b']) - float(bg_rgb[2]))
    px = np.clip([nr, ng, nb], 0, 255).astype(np.uint8).reshape(1, 1, 3)
    h, s, v = cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0]
    out = dict(feats)
    out.update({
        'mean_r': round(nr, 3),
        'mean_g': round(ng, 3),
        'mean_b': round(nb, 3),
        'hue': round(float(h), 3),
        'saturation': round(float(s), 3),
        'value': round(float(v), 3),
        'ratio_gr': round(ng / (nr + 1e-6), 4),
        'ratio_bg': round(nb / (ng + 1e-6), 4),
        'intensity': round(float(v), 3),
    })
    return out


def auto_roi(img_rgb, template_roi, ref_img_rgb=None, margin_px=40):
    """用模板 ROI 自动定位：固定机位直接映射；有参考图时用模板匹配微调。

    返回归一化 (x,y,w,h)。ref_img_rgb 提供模板 patch 的原始位置。
    """
    x, y, rw, rh = template_roi
    if ref_img_rgb is None:
        return (float(x), float(y), float(rw), float(rh))

    ref_h, ref_w = ref_img_rgb.shape[:2]
    img_h, img_w = img_rgb.shape[:2]
    rx0 = int(round(x * ref_w))
    ry0 = int(round(y * ref_h))
    rw_px = max(1, int(round(rw * ref_w)))
    rh_px = max(1, int(round(rh * ref_h)))
    rx1 = min(ref_w, rx0 + rw_px)
    ry1 = min(ref_h, ry0 + rh_px)
    if rx1 <= rx0 or ry1 <= ry0:
        return (float(x), float(y), float(rw), float(rh))

    patch = ref_img_rgb[ry0:ry1, rx0:rx1]
    patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # 若尺寸不同，将新图缩放到参考图尺度匹配
    work = img_rgb
    scale = 1.0
    if (img_h, img_w) != (ref_h, ref_w):
        work = cv2.resize(img_rgb, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
        scale = float(ref_w) / float(img_w)
    work_gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)

    margin = max(1, min(margin_px, ref_w // 8))
    # 搜索窗口 = 以模板原位置为中心、半径 margin 的完整滑动范围 + 模板宽度
    sx0 = max(0, rx0 - margin)
    sx1 = min(ref_w, rx0 + margin + patch.shape[1])
    if sx1 - sx0 < patch.shape[1] + 1:
        sx0 = max(0, sx1 - patch.shape[1] - 1)  # 贴边时向左补足
    sy0 = max(0, ry0 - margin)
    sy1 = min(ref_h, ry0 + margin + patch.shape[0])
    if sy1 - sy0 < patch.shape[0] + 1:
        sy0 = max(0, sy1 - patch.shape[0] - 1)
    if sx1 - sx0 < patch.shape[1] + 1 or sy1 - sy0 < patch.shape[0] + 1:
        return (float(x), float(y), float(rw), float(rh))

    res = cv2.matchTemplate(work_gray[sy0:sy1, sx0:sx1], patch_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.25:  # 匹配度过低视为未找到，直接映射
        return (float(x), float(y), float(rw), float(rh))

    best_x = sx0 + max_loc[0]
    best_y = sy0 + max_loc[1]
    off_x = (best_x - rx0) / ref_w
    off_y = (best_y - ry0) / ref_h
    nx = min(max(x + off_x, 0.0), 1.0 - rw)
    ny = min(max(y + off_y, 0.0), 1.0 - rh)
    return (float(nx), float(ny), float(rw), float(rh))
