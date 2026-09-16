# -*- coding: utf-8 -*-
"""合成测试图生成器（开发/验收辅助，非产品功能）。

模拟固定装置拍摄的荧光试纸：暗色背景 + 居中圆形检测区；
检测区颜色随已知浓度单调变化（浓度 0 → 绿色 hue≈120，浓度 max → 红色 hue≈0），
并叠加相机噪声，用于在真实试纸图提供前验证全流水线。

用法：
    python tools/make_test_images.py --out ../data/sample --conc 0,10,25,50,100 --reps 3
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def color_for_conc(conc, conc_max=100.0):
    """浓度 → OpenCV uint8 色相（范围 0~180）：
    conc=0 时 hue=60（绿，对应角度 120°），conc=conc_max 时 hue=0（红），线性插值。
    """
    hue = 60.0 * (1.0 - float(np.clip(conc, 0, conc_max)) / conc_max)
    return hue


def make_test_image(conc, size=(800, 600), conc_max=100.0, noise=6.0,
                    center=None, radius=None, background=(25, 25, 30)):
    """生成一张合成试纸图（返回 RGB ndarray）。

    检测区颜色为 HSV(hue, s=0.85, v=0.82)，hue 由 color_for_conc 决定。
    """
    h, w = size[1], size[0]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = np.array(background, dtype=np.uint8)

    cx, cy = center if center else (w // 2, h // 2)
    r = radius if radius else int(min(w, h) * 0.22)

    hue = color_for_conc(conc, conc_max)
    hsv = np.uint8([[[hue, 217, 209]]])  # s=0.85*255, v=0.82*255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0].astype(np.int16)

    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    img[mask] = np.clip(rgb, 0, 255).astype(np.uint8)

    # 相机噪声
    noise_arr = np.random.default_rng().normal(0, noise, img.shape)
    img = np.clip(img.astype(np.float64) + noise_arr, 0, 255).astype(np.uint8)
    return img


def make_batch(concs, out_dir, reps=1, size=(800, 600), prefix='strip'):
    """批量生成：每个浓度生成 reps 张，文件名含浓度信息。返回生成文件列表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for conc in concs:
        for i in range(reps):
            img = make_test_image(conc, size=size)
            name = f'{prefix}_c{conc:g}_r{i + 1}.png'
            path = out_dir / name
            cv2.imwrite(str(path), img[:, :, ::-1])  # RGB -> BGR
            files.append(str(path))
    return files


def main():
    ap = argparse.ArgumentParser(description='生成合成荧光试纸测试图')
    ap.add_argument('--out', default=str(ROOT / 'data' / 'sample'))
    ap.add_argument('--conc', default='0,10,25,50,100', help='逗号分隔浓度列表')
    ap.add_argument('--reps', type=int, default=1, help='每个浓度重复张数')
    ap.add_argument('--size', default='800x600', help='宽x高')
    args = ap.parse_args()

    concs = [float(x) for x in args.conc.split(',') if x.strip() != '']
    w, h = (int(x) for x in args.size.lower().split('x'))
    files = make_batch(concs, args.out, reps=args.reps, size=(w, h))
    print(f'已生成 {len(files)} 张合成图：')
    for f in files:
        print(' ', f)


if __name__ == '__main__':
    main()
