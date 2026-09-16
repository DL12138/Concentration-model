# -*- coding: utf-8 -*-
"""v1 功能改进测试集（问题 1~7b，逐步追加）。"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import create_app  # noqa: E402
from tools import make_test_images as mti  # noqa: E402

TEMPLATE = {'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44}


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    return app.test_client()


def _upload(client, conc=50, kind='detection'):
    img = mti.make_test_image(conc)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    assert ok
    data = {'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': kind}
    if kind == 'calibration':
        data['known_conc'] = str(conc)
    resp = client.post('/api/images/upload', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    return resp.get_json()['images'][0]['id']


def _pipeline(client, iids, with_template=True):
    if with_template:
        client.post('/api/templates', json=dict(TEMPLATE, name='T'))
    return client.post('/api/pipeline/run', json={'image_ids': iids}).get_json()


def _decode(resp):
    assert resp.status_code == 200
    assert resp.headers.get('Content-Type', '').startswith('image/png')
    arr = np.frombuffer(resp.data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB


# ============ 问题 1：每一步都加上处理后的图片 ============

def test_p1_processed_image_served(client):
    iid = _upload(client)
    _pipeline(client, [iid])
    r = client.get(f'/api/images/{iid}/processed')
    assert r.status_code == 200 and r.headers['Content-Type'].startswith('image/png')


def test_p1_overlay_image_with_roi(client):
    iid = _upload(client)
    _pipeline(client, [iid])
    r = client.get(f'/api/images/{iid}/overlay')
    img = _decode(r)
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_p1_overlay_without_roi_returns_source(client):
    iid = _upload(client)
    client.post('/api/pipeline/run', json={'image_ids': [iid]})  # 无模板 → 无 ROI
    r = client.get(f'/api/images/{iid}/overlay')
    assert r.status_code == 200


def test_p1_roi_crop_and_avg(client):
    iid = _upload(client, conc=50)
    _pipeline(client, [iid])
    crop = _decode(client.get(f'/api/images/{iid}/roi_crop'))
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    avg = _decode(client.get(f'/api/images/{iid}/roi_avg'))
    assert avg.shape == (128, 128, 3)
    # 平均色块 ≈ 特征表 mean（允许 ±6）
    feats = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    for i, key in enumerate(('mean_r', 'mean_g', 'mean_b')):
        assert abs(float(avg[0, 0, i]) - float(feats[key])) <= 6, f'{key} 平均色块不一致'


def test_p1_roi_crop_without_roi_404(client):
    iid = _upload(client)
    client.post('/api/pipeline/run', json={'image_ids': [iid]})
    assert client.get(f'/api/images/{iid}/roi_crop').status_code == 404
    assert client.get(f'/api/images/{iid}/roi_avg').status_code == 404
