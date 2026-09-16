# -*- coding: utf-8 -*-
"""M4 自动流水线编排测试：单图/批量执行、状态标记、images.status 更新、级联重算。"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import cv2  # noqa: E402

from app import create_app  # noqa: E402
from app import database as db  # noqa: E402
from tools import make_test_images as mti  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    return app.test_client()


def _upload(client, img=None, kind='detection', conc=None):
    if img is None:
        img = mti.make_test_image(25)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    assert ok
    data = {'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': kind, 'batch': 'B1'}
    if conc is not None:
        data['known_conc'] = str(conc)
    resp = client.post('/api/images/upload', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    return resp.get_json()['images'][0]['id']


def test_pipeline_without_template_marks_attention(client):
    iid = _upload(client)
    resp = client.post(f'/api/pipeline/{iid}/run', json={})
    assert resp.status_code == 200
    r = resp.get_json()['result']
    assert r['steps']['preprocess'] == 'ok'
    assert r['steps']['roi'] == 'attention'      # 无模板
    assert r['steps']['feature'] == 'attention'
    assert r['steps']['result'] == 'attention'   # 无生效模型
    assert r['status'] == 'attention'
    # images.status 更新
    row = db.query_one(Path(client.application.config['DATA_DIR']) / 'fluro.db',
                       'SELECT status FROM images WHERE id=?', (iid,))
    assert row['status'] == 'attention'


def test_pipeline_with_template_full_ok(client):
    _upload(client, mti.make_test_image(50), kind='calibration', conc=50)
    client.post('/api/templates', json={'name': 'T', 'x': 0.28, 'y': 0.28, 'w': 0.44, 'h': 0.44})
    # 保存生效模型（合成图语义 hue = 60 - 0.6*conc）
    client.post('/api/models', json={
        'name': 'M', 'type': 'linear', 'params': {'a': -0.6, 'b': 60},
        'metrics': {'r2': 0.999, 'rmse': 0.1, 'lod': 1.0},
        'source_snapshot': {'feature': 'hue', 'data': [[0, 60], [25, 45], [50, 30], [75, 15], [100, 0]]},
    })
    iid = _upload(client, mti.make_test_image(50))
    resp = client.post(f'/api/pipeline/{iid}/run', json={})
    r = resp.get_json()['result']
    assert r['steps']['preprocess'] == 'ok'
    assert r['steps']['roi'] == 'ok'
    assert r['steps']['feature'] == 'ok'
    assert r['steps']['result'] == 'ok'
    assert r['status'] == 'ok'
    # 特征已入库
    feats = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    assert feats is not None and feats['hue'] >= 0


def test_pipeline_batch(client):
    _upload(client, mti.make_test_image(25))
    _upload(client, mti.make_test_image(75))
    resp = client.post('/api/pipeline/run', json={})
    assert resp.status_code == 200
    results = resp.get_json()['results']
    assert len(results) == 2
    for r in results:
        assert r['steps']['preprocess'] == 'ok'


def test_pipeline_batch_explicit_ids(client):
    i1 = _upload(client)
    i2 = _upload(client)
    resp = client.post('/api/pipeline/run', json={'image_ids': [i2]})
    results = resp.get_json()['results']
    assert [r['image_id'] for r in results] == [i2]


def test_pipeline_only_processed_once_for_uploaded(client):
    """缺省 image_ids 时只处理 uploaded 状态的图。"""
    _upload(client)
    client.post('/api/pipeline/run', json={})
    resp = client.post('/api/pipeline/run', json={})
    assert resp.get_json()['results'] == []  # 已非 uploaded，不再处理


def test_pipeline_missing_image_handled(client):
    resp = client.post('/api/pipeline/99999/run', json={})
    assert resp.status_code == 200
    r = resp.get_json()['result']
    assert r['steps']['preprocess'] == 'error'
    assert r['status'] == 'error'


def test_preprocess_change_cascades_to_features(client):
    """改预处理参数重跑后，特征应基于新处理图（级联重算）。"""
    _upload(client, mti.make_test_image(50), kind='calibration', conc=50)
    client.post('/api/templates', json={'name': 'T', 'x': 0.28, 'y': 0.28, 'w': 0.44, 'h': 0.44})
    iid = _upload(client, mti.make_test_image(50))
    client.post(f'/api/pipeline/{iid}/run', json={})
    f1 = client.get(f'/api/pipeline/{iid}/features').get_json()['features']

    # 改用强中值滤波后重跑预处理 → 纹理熵应显著下降（级联重算生效）
    client.post(f'/api/pipeline/{iid}/preprocess', json={'filter': 'median', 'kernel': 15})
    f2 = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    assert f2['texture_entropy'] < f1['texture_entropy']


def test_roi_manual_fix_cascades(client):
    """手动修正 ROI 后特征自动重算。"""
    _upload(client, mti.make_test_image(50), kind='calibration', conc=50)
    client.post('/api/templates', json={'name': 'T', 'x': 0.28, 'y': 0.28, 'w': 0.44, 'h': 0.44})
    iid = _upload(client, mti.make_test_image(50))
    client.post(f'/api/pipeline/{iid}/run', json={})
    f1 = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    client.post(f'/api/pipeline/{iid}/roi', json={'x': 0.0, 'y': 0.0, 'w': 0.1, 'h': 0.1, 'source': 'manual'})
    f2 = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    assert f1['hue'] != f2['hue']
