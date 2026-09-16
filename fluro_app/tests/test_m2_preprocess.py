# -*- coding: utf-8 -*-
"""M2 预处理模块测试：算法正确性、接口、快照、参考图上传。"""
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
from app.image_processing import dark_flat_correct, denoise, preprocess  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    return app.test_client()


def _upload(client, img=None, kind='detection', batch='B1', conc=None):
    if img is None:
        img = np.full((200, 300, 3), 80, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', img)
    assert ok
    data = {'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': kind, 'batch': batch}
    if conc is not None:
        data['known_conc'] = str(conc)
    resp = client.post('/api/images/upload', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    return resp.get_json()['images'][0]['id']


# ---- 算法单元测试 ----

def test_dark_flat_correct_math():
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    dark = np.full((10, 10, 3), 100, dtype=np.uint8)
    flat = np.full((10, 10, 3), 200, dtype=np.uint8)
    out = dark_flat_correct(img, dark=dark, flat=flat)
    # (200-100)/(200-100) = 1.0 -> 255
    assert out.dtype == np.uint8
    assert np.allclose(out, 255)


def test_dark_flat_correct_flat_only():
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    flat = np.full((10, 10, 3), 200, dtype=np.uint8)
    out = dark_flat_correct(img, flat=flat)
    # 100/200 = 0.5 -> 127.5 -> 127/128
    assert np.allclose(out, 127, atol=1)


def test_dark_flat_correct_ref_resize():
    img = np.full((20, 30, 3), 150, dtype=np.uint8)
    dark = np.full((5, 5, 3), 50, dtype=np.uint8)  # 尺寸不同，应自动缩放
    out = dark_flat_correct(img, dark=dark)
    assert out.shape == img.shape
    assert np.allclose(out, 100, atol=1)


def test_denoise_shapes_and_smoothing():
    rng = np.random.default_rng(0)
    noisy = np.clip(rng.normal(128, 30, (100, 100, 3)), 0, 255).astype(np.uint8)
    for method in ('gaussian', 'median'):
        out = denoise(noisy, method=method, kernel=5)
        assert out.shape == noisy.shape
        assert out.std() < noisy.std(), f'{method} 去噪后应更平滑'
    # 偶数 kernel 自动取奇
    out = denoise(noisy, method='gaussian', kernel=4)
    assert out.shape == noisy.shape


def test_preprocess_combined():
    img = np.full((50, 50, 3), 180, dtype=np.uint8)
    dark = np.full((50, 50, 3), 30, dtype=np.uint8)
    flat = np.full((50, 50, 3), 220, dtype=np.uint8)
    out = preprocess(img, method='median', kernel=3, dark=dark, flat=flat)
    assert out.shape == img.shape
    # (180-30)/(220-30)=0.789 -> ~201
    assert np.allclose(out, 201, atol=2)


# ---- 接口测试 ----

def test_preprocess_api_returns_processed(client, tmp_path):
    # 生成参考图文件
    ref_dir = Path(tmp_path) / 'refs'
    ref_dir.mkdir(exist_ok=True)
    dark_p = ref_dir / 'dark.png'
    flat_p = ref_dir / 'flat.png'
    cv2.imwrite(str(dark_p), np.full((200, 300, 3), 20, dtype=np.uint8)[:, :, ::-1])
    cv2.imwrite(str(flat_p), np.full((200, 300, 3), 240, dtype=np.uint8)[:, :, ::-1])

    iid = _upload(client)
    resp = client.post(f'/api/pipeline/{iid}/preprocess', json={
        'filter': 'gaussian', 'kernel': 5,
        'use_dark': True, 'use_flat': True,
        'dark_path': str(dark_p), 'flat_path': str(flat_p),
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert body['params']['filter'] == 'gaussian'
    assert body['processed_url'].endswith('/processed')

    r = client.get(body['processed_url'])
    assert r.status_code == 200
    assert r.mimetype.startswith('image/')


def test_preprocess_snapshot_written(client, tmp_path):
    iid = _upload(client)
    client.post(f'/api/pipeline/{iid}/preprocess', json={'filter': 'median', 'kernel': 3})
    step = db.query_one(Path(tmp_path) / 'fluro.db',
                        'SELECT * FROM pipeline_steps WHERE image_id=? AND step=?', (iid, 'preprocess'))
    assert step is not None
    assert step['status'] == 'ok'
    params = json.loads(step['params_json'])
    assert params['filter'] == 'median' and params['kernel'] == 3


def test_preprocess_snapshot_updated(client, tmp_path):
    iid = _upload(client)
    client.post(f'/api/pipeline/{iid}/preprocess', json={'filter': 'median', 'kernel': 3})
    client.post(f'/api/pipeline/{iid}/preprocess', json={'filter': 'gaussian', 'kernel': 7})
    rows = db.query(Path(tmp_path) / 'fluro.db',
                    'SELECT * FROM pipeline_steps WHERE image_id=? AND step=?', (iid, 'preprocess'))
    assert len(rows) == 1, '同一图同一步骤只保留一条快照（更新而非新增）'
    assert json.loads(rows[0]['params_json'])['filter'] == 'gaussian'


def test_preprocess_invalid_filter_falls_back(client):
    iid = _upload(client)
    resp = client.post(f'/api/pipeline/{iid}/preprocess', json={'filter': 'bogus', 'kernel': 'x'})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['params']['filter'] == 'gaussian'
    assert body['params']['kernel'] == 5


def test_preprocess_missing_image_404(client):
    resp = client.post('/api/pipeline/99999/preprocess', json={})
    assert resp.status_code == 404


def test_ref_upload_and_settings(client, tmp_path):
    img = np.full((100, 100, 3), 10, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', img)
    assert ok
    resp = client.post('/api/refs/dark', data={
        'file': (io.BytesIO(buf.tobytes()), 'dark.png'),
    }, content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    path = body['path']
    assert Path(path).exists()
    saved = db.get_setting(Path(tmp_path) / 'fluro.db', 'dark_ref_path')
    assert saved == path

    resp = client.get('/api/refs')
    refs = resp.get_json()
    assert refs['dark_path'] == path


def test_ref_upload_invalid_kind(client):
    resp = client.post('/api/refs/bogus', data={}, content_type='multipart/form-data')
    assert resp.status_code == 400


def test_processed_404_without_run(client):
    iid = _upload(client)
    r = client.get(f'/api/images/{iid}/processed')
    assert r.status_code == 404
