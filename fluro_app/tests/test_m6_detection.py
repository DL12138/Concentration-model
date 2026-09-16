# -*- coding: utf-8 -*-
"""M6 浓度检测与判定测试：C±U 预测、超限判定、快照可追溯、级联重算、记录检索。"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import cv2  # noqa: E402

from app import create_app  # noqa: E402
from app import database as db  # noqa: E402
from app.routes.api_model import _judge  # noqa: E402
from tools import make_test_images as mti  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    return app.test_client()


def _upload(client, img=None, kind='detection', conc=None, batch='B1'):
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


def _make_active_model(client):
    """保存一个生效模型（合成图语义 hue = 60 - 0.6*conc；数据带微小测量噪声）。"""
    client.post('/api/models', json={
        'name': 'M', 'type': 'linear', 'params': {'a': -0.6, 'b': 60},
        'metrics': {'r2': 0.999, 'rmse': 0.5, 'lod': 1.0},
        'source_snapshot': {
            'feature': 'hue',
            'data': [[0, 61], [25, 44], [50, 31], [75, 14], [100, 1]],
        },
    })


def _setup_full(client, conc=50):
    """模板 + 生效模型 + 上传检测图并跑流水线。"""
    _upload(client, mti.make_test_image(50), kind='calibration', conc=50)
    client.post('/api/templates', json={'name': 'T', 'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44})
    _make_active_model(client)
    iid = _upload(client, mti.make_test_image(conc))
    client.post(f'/api/pipeline/{iid}/run', json={})
    return iid


def test_detect_conc_accuracy(client):
    """检测图 conc=50 → 检测结果应接近 50。"""
    iid = _setup_full(client, conc=50)
    resp = client.post(f'/api/detect/{iid}', json={})
    assert resp.status_code == 200
    d = resp.get_json()['detection']
    assert abs(d['conc'] - 50) < 8
    assert d['u'] > 0
    assert d['status'] == 'within'
    assert d['model_name'] == 'M'


def test_pipeline_runs_detection_automatically(client):
    """流水线后 result 步骤自动完成，detections 表有记录。"""
    iid = _setup_full(client, conc=75)
    row = db.query_one(Path(client.application.config['DATA_DIR']) / 'fluro.db',
                       'SELECT * FROM detections WHERE image_id=?', (iid,))
    assert row is not None
    assert abs(row['conc'] - 75) < 10
    # 快照完整可追溯
    snap = __import__('json').loads(row['params_snapshot_json'])
    assert snap['model_id'] and snap['feature'] == 'hue'
    assert snap['calibration_n'] == 5 and len(snap['calibration_data']) == 5
    assert snap['model_params']['a'] == -0.6
    assert 'limits' in snap and snap['conf'] == 0.95


def test_detect_requires_model(client):
    """特征已就绪但无生效模型 → 报模型错误。"""
    _upload(client, mti.make_test_image(50), kind='calibration', conc=50)
    client.post('/api/templates', json={'name': 'T', 'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44})
    iid = _upload(client)
    client.post(f'/api/pipeline/{iid}/run', json={})
    resp = client.post(f'/api/detect/{iid}', json={})
    assert resp.status_code == 400
    assert '模型' in resp.get_json()['error']


def test_detect_requires_features(client):
    """未跑流水线（无特征）的检测图应报错。"""
    _make_active_model(client)
    iid = _upload(client)
    resp = client.post(f'/api/detect/{iid}', json={})
    assert resp.status_code == 400
    assert '特征' in resp.get_json()['error']


def test_detect_rejects_calibration_image(client):
    """标定图不能直接检测。"""
    _make_active_model(client)
    iid = _upload(client, mti.make_test_image(50), kind='calibration', conc=50)
    client.post('/api/templates', json={'name': 'T', 'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44})
    client.post(f'/api/pipeline/{iid}/run', json={})
    resp = client.post(f'/api/detect/{iid}', json={})
    assert resp.status_code == 400


def test_judge_logic():
    """超限判定逻辑：within/above/below/borderline。"""
    assert _judge(50, 5, 0, 100) == 'within'
    assert _judge(150, 5, 0, 100) == 'above'
    assert _judge(-10, 5, 0, 100) == 'below'
    assert _judge(98, 5, 0, 100) == 'borderline'   # C-U=93<100? no -> C+U=103>100 -> borderline
    assert _judge(2, 5, 0, 100) == 'borderline'    # C-U=-3<0 -> borderline
    assert _judge(50, 5, None, None) == 'within'   # 无上下限
    assert _judge(-5, 1, 0, None) == 'below'       # 仅下限


def test_detect_with_limits(client):
    """设置上下限后检测，判定应生效（用 hue 稳定区 conc=50）。"""
    iid = _setup_full(client, conc=50)   # 检测约 50
    db.set_setting(Path(client.application.config['DATA_DIR']) / 'fluro.db', 'limit_lower', '0')
    db.set_setting(Path(client.application.config['DATA_DIR']) / 'fluro.db', 'limit_upper', '40')
    d = client.post(f'/api/detect/{iid}', json={}).get_json()['detection']
    assert d['status'] in ('above', 'borderline')
    # 下限高于检测值 → below
    db.set_setting(Path(client.application.config['DATA_DIR']) / 'fluro.db', 'limit_lower', '70')
    db.set_setting(Path(client.application.config['DATA_DIR']) / 'fluro.db', 'limit_upper', None)
    d = client.post(f'/api/detect/{iid}', json={}).get_json()['detection']
    assert d['status'] in ('below', 'borderline')


def test_cascade_recompute_detection(client):
    """改 ROI 后检测结果应重算（级联）。"""
    iid = _setup_full(client, conc=50)
    d1 = client.post(f'/api/detect/{iid}', json={}).get_json()['detection']
    # 手动改 ROI 到角落（特征变化）→ 触发 recompute_downstream → 检测更新
    client.post(f'/api/pipeline/{iid}/roi', json={'x': 0.0, 'y': 0.0, 'w': 0.1, 'h': 0.1, 'source': 'manual'})
    d2 = client.post(f'/api/detect/{iid}', json={}).get_json()['detection']
    assert d1['conc'] != d2['conc']


def test_detections_search(client):
    """记录检索：按批次与关键词。"""
    _setup_full(client, conc=50)   # batch B1
    rows = client.get('/api/detections').get_json()['detections']
    assert len(rows) == 1
    rows = client.get('/api/detections?batch=B1').get_json()['detections']
    assert len(rows) == 1
    rows = client.get('/api/detections?batch=XX').get_json()['detections']
    assert len(rows) == 0
    rows = client.get('/api/detections?q=within').get_json()['detections']
    assert len(rows) == 1
