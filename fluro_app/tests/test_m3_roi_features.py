# -*- coding: utf-8 -*-
"""M3 ROI 与特征提取测试：特征算法、ROI 自动套用、模板 CRUD、接口与级联。"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import create_app  # noqa: E402
from app import database as db  # noqa: E402
from app.image_processing import (extract_features, auto_roi, crop_roi,
                                  roi_to_pixels, preprocess)  # noqa: E402
from tools import make_test_images as mti  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    return app.test_client()


def _upload(client, img=None, kind='detection', batch='B1', conc=None):
    if img is None:
        img = mti.make_test_image(25)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    assert ok
    data = {'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': kind, 'batch': batch}
    if conc is not None:
        data['known_conc'] = str(conc)
    resp = client.post('/api/images/upload', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    return resp.get_json()['images'][0]['id']


# ---- 算法单元测试 ----

def test_roi_to_pixels():
    shape = (100, 200, 3)
    assert roi_to_pixels((0.25, 0.5, 0.5, 0.25), shape) == (50, 50, 150, 75)
    # 越界裁剪
    x0, y0, x1, y1 = roi_to_pixels((0.9, 0.9, 0.5, 0.5), shape)
    assert x1 <= 200 and y1 <= 100


def test_crop_roi_empty_raises():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_roi(img, (0, 0, 0, 0.1))


def test_extract_features_known_color():
    # 纯色图 + 全图 ROI：特征应为该颜色
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (255, 0, 0)  # R
    f = extract_features(img, (0, 0, 1, 1))
    assert f['mean_r'] > 250 and f['mean_g'] < 5 and f['mean_b'] < 5
    assert f['ratio_gr'] < 0.01
    # 纹理熵：纯色图接近 0
    assert f['texture_entropy'] < 0.5


def test_extract_features_uses_roi():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = (0, 255, 0)  # 中央绿色
    f = extract_features(img, (0.25, 0.25, 0.5, 0.5))
    assert f['mean_g'] > 240 and f['mean_r'] < 15
    # 框在角落：应取背景
    f2 = extract_features(img, (0, 0, 0.1, 0.1))
    assert f2['mean_g'] < 15


def test_extract_features_values_present():
    img = mti.make_test_image(50)
    f = extract_features(img, (0.3, 0.3, 0.4, 0.4))
    for key in ('mean_r', 'mean_g', 'mean_b', 'hue', 'saturation', 'value',
                'ratio_gr', 'ratio_bg', 'intensity', 'texture_entropy'):
        assert key in f and f[key] is not None


def test_auto_roi_direct_mapping():
    tpl = (0.2, 0.3, 0.4, 0.2)
    img = mti.make_test_image(10)
    out = auto_roi(img, tpl)
    assert out == pytest.approx(tpl)


def test_auto_roi_shift_correction():
    """参考图中检测区在 (0.28,0.28,0.44,0.44)；新图中检测区整体偏移 +40px，
    模板匹配应把 ROI 修正到 (0.38,0.38,...)。"""
    size = (400, 400)
    ref = mti.make_test_image(50, size=size)   # 检测圆圆心 (200,200)，r=88
    shifted = np.full((400, 400, 3), 25, dtype=np.uint8)
    yy, xx = np.mgrid[0:400, 0:400]
    mask = (xx - 240) ** 2 + (yy - 240) ** 2 <= 88 ** 2  # 圆心移到 (240,240)
    hue = mti.color_for_conc(50)
    hsv = np.uint8([[[hue, 217, 209]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    shifted[mask] = rgb

    tpl = (0.28, 0.28, 0.44, 0.44)  # 模板紧贴参考图检测区
    out = auto_roi(shifted, tpl, ref_img_rgb=ref, margin_px=60)
    assert abs(out[0] - 0.38) < 0.06, f'x 应修正到 ~0.38，得到 {out[0]}'
    assert abs(out[1] - 0.38) < 0.06, f'y 应修正到 ~0.38，得到 {out[1]}'


# ---- 接口测试 ----

def test_template_crud_and_activate(client):
    tid = client.post('/api/templates', json={
        'name': 'T1', 'x': 0.2, 'y': 0.3, 'w': 0.4, 'h': 0.2, 'ref_image_id': None,
    }).get_json()['id']
    rows = client.get('/api/templates').get_json()['templates']
    assert len(rows) == 1 and rows[0]['is_active'] == 1  # 首个模板自动激活

    tid2 = client.post('/api/templates', json={
        'name': 'T2', 'x': 0.1, 'y': 0.1, 'w': 0.3, 'h': 0.3, 'ref_image_id': None,
    }).get_json()['id']
    client.post(f'/api/templates/{tid2}/activate', json={})
    rows = client.get('/api/templates').get_json()['templates']
    active = [r for r in rows if r['is_active']]
    assert len(active) == 1 and active[0]['id'] == tid2

    client.delete(f'/api/templates/{tid2}')
    rows = client.get('/api/templates').get_json()['templates']
    assert len(rows) == 1 and rows[0]['is_active'] == 1  # 删除后自动补激活


def test_template_invalid_roi_rejected(client):
    resp = client.post('/api/templates', json={'name': 'X', 'x': 0.9, 'y': 0.9, 'w': 0.5, 'h': 0.5})
    assert resp.status_code == 400


def test_auto_roi_api_uses_active_template(client):
    _upload(client, mti.make_test_image(50, size=(400, 400)), kind='calibration', conc=50)
    client.post('/api/templates', json={'name': 'T', 'x': 0.2, 'y': 0.3, 'w': 0.4, 'h': 0.2, 'ref_image_id': None})
    iid = _upload(client, mti.make_test_image(50, size=(400, 400)))
    resp = client.post(f'/api/pipeline/{iid}/roi/auto', json={})
    assert resp.status_code == 200
    roi = resp.get_json()['roi']
    assert abs(roi['x'] - 0.2) < 0.02 and abs(roi['y'] - 0.3) < 0.02
    assert roi['source'] == 'auto'


def test_auto_roi_without_template_errors(client):
    iid = _upload(client)
    resp = client.post(f'/api/pipeline/{iid}/roi/auto', json={})
    assert resp.status_code == 400


def test_manual_roi_save_and_get(client):
    iid = _upload(client)
    resp = client.post(f'/api/pipeline/{iid}/roi', json={'x': 0.1, 'y': 0.2, 'w': 0.3, 'h': 0.4, 'source': 'manual'})
    assert resp.status_code == 200
    got = client.get(f'/api/pipeline/{iid}/roi').get_json()['roi']
    assert got['source'] == 'manual' and got['x'] == 0.1


def test_features_api_requires_roi(client):
    iid = _upload(client)
    resp = client.post(f'/api/pipeline/{iid}/features', json={})
    assert resp.status_code == 400
    assert 'ROI' in resp.get_json()['error']


def test_features_api_computes_and_stores(client):
    iid = _upload(client, mti.make_test_image(50))
    client.post(f'/api/pipeline/{iid}/roi', json={'x': 0.3, 'y': 0.3, 'w': 0.4, 'h': 0.4, 'source': 'manual'})
    resp = client.post(f'/api/pipeline/{iid}/features', json={})
    assert resp.status_code == 200
    feats = resp.get_json()['features']
    assert feats['mean_r'] >= 0 and feats['hue'] >= 0

    got = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    assert got['hue'] == feats['hue']
    # 快照写入
    step = db.query_one(Path(client.application.config['DATA_DIR']) / 'fluro.db',
                        'SELECT * FROM pipeline_steps WHERE image_id=? AND step=?', (iid, 'feature'))
    assert step and step['status'] == 'ok'


def test_roi_change_cascades_features(client):
    """ROI 修正后特征应自动重算（级联）。"""
    iid = _upload(client, mti.make_test_image(50))
    client.post(f'/api/pipeline/{iid}/roi', json={'x': 0.3, 'y': 0.3, 'w': 0.4, 'h': 0.4, 'source': 'manual'})
    f1 = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    # 换成角落小框
    client.post(f'/api/pipeline/{iid}/roi', json={'x': 0.0, 'y': 0.0, 'w': 0.1, 'h': 0.1, 'source': 'manual'})
    f2 = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    assert f1['hue'] != f2['hue']


def test_preprocess_then_features_uses_processed(client):
    """特征应基于处理后图像计算（存在 processed 时）。"""
    iid = _upload(client, mti.make_test_image(50))
    client.post(f'/api/pipeline/{iid}/preprocess', json={'filter': 'median', 'kernel': 7})
    client.post(f'/api/pipeline/{iid}/roi', json={'x': 0.3, 'y': 0.3, 'w': 0.4, 'h': 0.4, 'source': 'manual'})
    resp = client.post(f'/api/pipeline/{iid}/features', json={})
    assert resp.status_code == 200
