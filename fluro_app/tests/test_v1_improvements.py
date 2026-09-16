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
    # 纯灰图（无亮/饱和检测区）：自动识别失败，无 ROI → crop/avg 返回 404
    img = np.full((300, 400, 3), 90, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    data = {'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': 'detection'}
    iid = client.post('/api/images/upload', data=data, content_type='multipart/form-data').get_json()['images'][0]['id']
    client.post('/api/pipeline/run', json={'image_ids': [iid]})
    assert client.get(f'/api/images/{iid}/roi_crop').status_code == 404
    assert client.get(f'/api/images/{iid}/roi_avg').status_code == 404


# ============ 问题 2：上传删除按钮 + 自动下一步（不强制人工） ============

def test_p2_delete_image_cleans_all(client):
    iid = _upload(client, conc=20, kind='calibration')
    _pipeline(client, [iid])
    # 产生检测记录：先建生效模型再检测
    g = client.post('/api/calibration/groups', json={'conc': 20}).get_json()
    client.post('/api/calibration/points', json={'image_id': iid, 'group_id': g['id']})
    client.post('/api/calibration/fit', json={'feature': 'hue'})
    client.post('/api/models', json={'name': 'M', 'type': 'linear', 'params': {'a': 1, 'b': 0},
                                     'metrics': {'r2': 0.99, 'rmse': 1, 'lod': 1}, 'source_snapshot': {}})
    client.post('/api/detect/' + str(iid), json={})
    img_info = client.get(f'/api/images/{iid}').get_json()
    proc = Path(client.application.config['DATA_DIR']) / 'processed' / f'{iid}.png'
    assert proc.exists() and Path(img_info['file_path']).exists()

    resp = client.delete(f'/api/images/{iid}')
    assert resp.status_code == 200 and resp.get_json()['ok'] is True
    assert client.get(f'/api/images/{iid}').status_code == 404
    assert client.get(f'/api/pipeline/{iid}/roi').status_code == 404  # roi 已清理
    assert client.get(f'/api/pipeline/{iid}/features').get_json()['features'] is None
    assert not proc.exists() and not Path(img_info['file_path']).exists()


def test_p2_delete_missing_404(client):
    assert client.delete('/api/images/9999').status_code == 404


def test_p2_rerun_pipeline_all(client):
    """「自动处理全部」= 全量重跑流水线，不需要人工逐步处理。"""
    iids = [_upload(client, conc=c) for c in (10, 50)]
    _pipeline(client, iids, with_template=False)
    # 无模板：ROI 标记 attention 待人工，但重跑流水线本身应返回 ok
    res = client.post('/api/pipeline/run', json={'image_ids': iids}).get_json()
    assert res.get('ok') is True
    for iid in iids:
        st = client.get(f'/api/images/{iid}').get_json()['status']
        assert st in ('ok', 'attention', 'processing')


# ============ 问题 3：暗场/平场参考图预览 + 前后对比效果 ============

def _upload_ref(client, kind, color=0):
    img = np.full((120, 160, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    data = {'file': (io.BytesIO(buf.tobytes()), 'ref.png')}
    return client.post(f'/api/refs/{kind}', data=data, content_type='multipart/form-data')


def test_p3_ref_upload_and_preview_image(client):
    r = _upload_ref(client, 'dark', color=30)
    assert r.status_code == 200 and r.get_json()['ok'] is True
    img = _decode(client.get('/api/refs/dark/image'))
    assert img.shape == (120, 160, 3) and int(img[0, 0, 0]) == 30
    r2 = _upload_ref(client, 'flat', color=200)
    assert r2.status_code == 200
    assert _decode(client.get('/api/refs/flat/image')).shape == (120, 160, 3)


def test_p3_ref_not_set_404_and_clear(client):
    assert client.get('/api/refs/dark/image').status_code == 404
    _upload_ref(client, 'dark')
    assert client.get('/api/refs/dark/image').status_code == 200
    assert client.delete('/api/refs/dark').get_json()['ok'] is True
    assert client.get('/api/refs/dark/image').status_code == 404


def test_p3_ref_changes_processed_image(client):
    """前后对比：同一张图在有无暗场参考图两种设置下，处理后图像不同。"""
    iid = _upload(client)
    _pipeline(client, [iid])
    client.post('/api/pipeline/' + str(iid) + '/preprocess',
                json={'filter': 'gaussian', 'kernel': 5, 'use_dark': False, 'use_flat': False})
    r1 = client.get(f'/api/images/{iid}/processed')
    before = _decode(r1)

    _upload_ref(client, 'dark', color=40)
    client.post('/api/pipeline/' + str(iid) + '/preprocess',
                json={'filter': 'gaussian', 'kernel': 5, 'use_dark': True, 'use_flat': False})
    r2 = client.get(f'/api/images/{iid}/processed')
    after = _decode(r2)

    diff = np.abs(before.astype(int) - after.astype(int))
    assert diff.mean() > 0.5, '暗场校正未改变处理结果（无前后对比差异）'


# ============ 问题 4：ROI 自动识别 + 修正图 + 背景扣除 ============

def test_p4_auto_detect_without_template(client):
    """无模板时基于图像内容自动识别检测区（不强制人工框选）。"""
    iid = _upload(client, conc=50)
    res = client.post('/api/pipeline/run', json={'image_ids': [iid]}).get_json()
    assert res['results'][0]['steps']['roi'] == 'ok'
    roi = client.get(f'/api/pipeline/{iid}/roi').get_json()['roi']
    assert roi and roi['source'] == 'auto'
    # ROI 应覆盖合成图检测圆（圆心约在 (0.5, 0.5)）
    cx = roi['x'] + roi['w'] / 2
    cy = roi['y'] + roi['h'] / 2
    assert 0.42 <= cx <= 0.58 and 0.42 <= cy <= 0.58


def test_p4_roi_bg_subtract_field(client):
    iid = _upload(client, conc=50)
    _pipeline(client, [iid])
    r = client.post(f'/api/pipeline/{iid}/roi', json=dict(TEMPLATE, source='manual', bg_subtract=1))
    assert r.get_json()['roi']['bg_subtract'] == 1
    roi = client.get(f'/api/pipeline/{iid}/roi').get_json()['roi']
    assert roi['bg_subtract'] == 1


def test_p4_bg_subtraction_changes_features(client):
    """开启背景扣除后特征均值应低于未扣除（合成图背景暗于检测区）。"""
    iid = _upload(client, conc=100)  # 红色亮区
    _pipeline(client, [iid])
    base = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    client.post(f'/api/pipeline/{iid}/roi', json=dict(TEMPLATE, source='manual', bg_subtract=1))
    sub = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    for key in ('mean_r', 'mean_g', 'mean_b'):
        assert sub[key] <= base[key] + 1e-6, f'{key} 背景扣除后未下降'


def test_p4_overlay_shows_after_manual_save(client):
    """手动修正保存后，overlay（修正后图）接口可访问。"""
    iid = _upload(client, conc=50)
    _pipeline(client, [iid])
    client.post(f'/api/pipeline/{iid}/roi', json=dict(TEMPLATE, source='manual', bg_subtract=0))
    assert client.get(f'/api/images/{iid}/overlay').status_code == 200


# ============ 问题 5：RGB 通道分离流程界面 ============

def test_p5_channels_compute_and_get(client):
    iid = _upload(client, conc=50)
    _pipeline(client, [iid])
    r = client.post(f'/api/pipeline/{iid}/channels', json={})
    assert r.status_code == 200
    c = r.get_json()['channels']
    assert c['r_url'] and c['g_url'] and c['b_url']
    assert abs(c['mean_r'] - 50) < 60  # 合成图 R 通道有值（绿图 R≈50）
    got = client.get(f'/api/pipeline/{iid}/channels').get_json()['channels']
    assert got['mean_g'] == c['mean_g']


def test_p5_channel_images_served(client):
    iid = _upload(client, conc=100)
    _pipeline(client, [iid])
    client.post(f'/api/pipeline/{iid}/channels', json={})
    for ch in 'rgb':
        r = client.get(f'/api/images/{iid}/channel/{ch}')
        assert r.status_code == 200 and r.headers['Content-Type'].startswith('image/png')
        gray = _decode(r)
        assert gray.shape[2] == 3
    assert client.get(f'/api/images/{iid}/channel/x').status_code == 400


def test_p5_pipeline_includes_channels_step(client):
    iid = _upload(client, conc=50)
    res = client.post('/api/pipeline/run', json={'image_ids': [iid]}).get_json()
    assert res['results'][0]['steps']['channels'] == 'ok'
    assert client.get(f'/api/pipeline/{iid}/channels').get_json()['channels'] is not None


def test_p5_delete_cleans_channel_files(client):
    iid = _upload(client, conc=50)
    _pipeline(client, [iid])
    client.post(f'/api/pipeline/{iid}/channels', json={})
    cdir = Path(client.application.config['DATA_DIR']) / 'processed' / 'channels'
    assert (cdir / f'{iid}_r.png').exists()
    client.delete(f'/api/images/{iid}')
    assert not (cdir / f'{iid}_r.png').exists()
