# -*- coding: utf-8 -*-
"""M7 首页聚合与检测记录测试：summary 字段、记录检索、CSV 导出。"""
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import cv2  # noqa: E402

from app import create_app  # noqa: E402
from tools import make_test_images as mti  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    return app.test_client()


def _upload(client, conc=None, kind='detection', batch='B1'):
    img = mti.make_test_image(conc if conc is not None else 50)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    assert ok
    data = {'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': kind, 'batch': batch}
    if conc is not None:
        data['known_conc'] = str(conc)
    resp = client.post('/api/images/upload', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    return resp.get_json()['images'][0]['id']


def _make_full_env(client):
    """模板 + 生效模型 + 标定分组 + 一张检测图完成流水线。返回检测图 id。"""
    cal_id = _upload(client, conc=50, kind='calibration')
    client.post('/api/templates', json={'name': 'T', 'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44})
    g = client.post('/api/calibration/groups', json={'conc': 50}).get_json()
    client.post('/api/calibration/points', json={'image_id': cal_id, 'group_id': g['id']})
    client.post('/api/models', json={
        'name': 'M', 'type': 'linear', 'params': {'a': -0.6, 'b': 60},
        'metrics': {'r2': 0.999, 'rmse': 0.5, 'lod': 1.0},
        'source_snapshot': {'feature': 'hue', 'data': [[0, 61], [25, 44], [50, 31], [75, 14], [100, 1]]},
    })
    iid = _upload(client, batch='B7')
    client.post(f'/api/pipeline/{iid}/run', json={})
    return iid


def test_summary_empty_state(client):
    s = client.get('/api/home/summary').get_json()
    assert s['model_name'] is None
    assert s['calibrated'] is False
    assert s['last_detection'] is None
    assert s['today_count'] == 0
    assert s['data_dir']


def test_summary_after_detection(client):
    iid = _make_full_env(client)
    s = client.get('/api/home/summary').get_json()
    assert s['model_name'] == 'M'
    assert s['model_r2'] == 0.999
    assert s['calibrated'] is True
    assert s['n_groups'] >= 1
    assert s['last_detection'] is not None
    assert s['last_detection']['image_id'] == iid
    assert s['last_detection']['conc'] is not None
    assert s['today_count'] >= 2   # 标定图 + 检测图
    assert s['total_detections'] >= 1


def test_records_search_by_batch_and_q(client):
    _make_full_env(client)
    rows = client.get('/api/detections?batch=B7').get_json()['detections']
    assert len(rows) == 1
    rows = client.get('/api/detections?batch=NOPE').get_json()['detections']
    assert rows == []
    rows = client.get('/api/detections?q=within').get_json()['detections']
    assert len(rows) == 1
    rows = client.get('/api/detections?q=M').get_json()['detections']
    assert rows == []  # 关键词仅匹配批次/判定


def test_export_csv(client):
    _make_full_env(client)
    resp = client.get('/api/detections/export')
    assert resp.status_code == 200
    assert 'text/csv' in resp.mimetype
    lines = resp.data.decode('utf-8').strip().splitlines()
    reader = list(csv.reader(io.StringIO(resp.data.decode('utf-8'))))
    assert reader[0] == ['ID', '图片ID', '批次', '浓度C', '不确定度U', '判定', '检测时间']
    assert len(reader) == 2  # 表头 + 1 行
    assert reader[1][2] == 'B7'
    assert reader[1][5] == 'within'
    # 带筛选导出
    resp2 = client.get('/api/detections/export?batch=NOPE')
    reader2 = list(csv.reader(io.StringIO(resp2.data.decode('utf-8'))))
    assert len(reader2) == 1  # 仅表头
