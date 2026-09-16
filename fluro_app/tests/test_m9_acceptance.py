# -*- coding: utf-8 -*-
"""M9 验收测试：启动冒烟、数据持久化、性能基线、空状态、友好错误、全链路端到端。

对应 PRD v1.1 第 11 章完成定义的 A~G 验收项与性能基线。
"""
import io
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import cv2  # noqa: E402

from app import create_app  # noqa: E402
from app import database as db  # noqa: E402
from tools import make_test_images as mti  # noqa: E402

VENV_PY = ROOT / '.venv' / 'Scripts' / 'python.exe'


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    return app.test_client()


def _upload(client, conc, kind='calibration', batch='B9'):
    img = mti.make_test_image(conc)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    assert ok
    data = {'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': kind, 'batch': batch,
            'known_conc': str(conc)}
    resp = client.post('/api/images/upload', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    return resp.get_json()['images'][0]['id']


def _build_env(client):
    """标定 5 浓度 + 模板 + 分组 + 模型 + 检测图流水线。返回检测图 id。"""
    ids = [_upload(client, c) for c in (0, 25, 50, 75, 100)]
    client.post('/api/templates', json={'name': 'T', 'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44})
    client.post('/api/pipeline/run', json={'image_ids': ids})   # 先完成特征
    for c, iid in zip((0, 25, 50, 75, 100), ids):
        g = client.post('/api/calibration/groups', json={'conc': c}).get_json()
        assert g.get('id'), f'建组失败：{g}'
        p = client.post('/api/calibration/points', json={'image_id': iid, 'group_id': g['id']})
        assert p.status_code == 200, p.get_json()
    j = client.post('/api/calibration/fit', json={'feature': 'ratio_gr'}).get_json()
    assert j.get('ok'), f'拟合失败：{j}'
    best = j['best']
    assert best, '无可用模型'
    assert 'error' not in j['results'][best]
    r = j['results'][best]
    client.post('/api/models', json={
        'name': 'M9', 'type': best, 'params': r['params'],
        'metrics': {'r2': r['r2'], 'rmse': r['rmse'], 'lod': r['lod']},
        'source_snapshot': {'feature': 'ratio_gr', 'data': j['data']},
    })
    det_id = _upload(client, 50, kind='detection', batch='B9D')
    client.post(f'/api/pipeline/{det_id}/run', json={})
    return det_id


# ---------- A：启动即用（真实进程冒烟） ----------

def test_startup_smoke_real_process(tmp_path):
    """真实拉起 server.py 子进程，访问首页与静态资源。"""
    env = dict(os.environ)
    env['FLURO_PORT'] = '8799'
    env['FLURO_NO_BROWSER'] = '1'
    env['FLURO_DATA_DIR'] = str(tmp_path)
    proc = subprocess.Popen([str(VENV_PY), 'server.py'], cwd=str(ROOT),
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        import urllib.request
        ok_home = ok_js = False
        for _ in range(40):
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen('http://127.0.0.1:8799/', timeout=2) as r:
                    ok_home = r.status == 200
                with urllib.request.urlopen('http://127.0.0.1:8799/static/js/views/home.js', timeout=2) as r:
                    ok_js = r.status == 200
                if ok_home and ok_js:
                    break
            except Exception:
                time.sleep(0.3)
        assert ok_home and ok_js, '真实进程未能提供首页/静态资源'
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------- B：数据持久化（重启不丢） ----------

def test_data_persists_across_restart(tmp_path):
    data_dir = str(tmp_path)
    app1 = create_app({'TESTING': True, 'DATA_DIR': data_dir})
    c1 = app1.test_client()
    iid = _upload(c1, 50, kind='detection', batch='B9')
    # 模拟关闭：销毁 app1 连接，重新 create_app（等价于重启进程读同一磁盘数据）
    app2 = create_app({'TESTING': True, 'DATA_DIR': data_dir})
    c2 = app2.test_client()
    rows = db.query(Path(data_dir) / 'fluro.db', 'SELECT * FROM images ORDER BY id')
    assert any(r['id'] == iid for r in rows)
    j = c2.get('/api/images').get_json()
    assert len(j['images']) == 1


# ---------- C/D/E/F/G：功能验收（全链路一镜到底） ----------

def test_full_workflow_e2e(client):
    """标定 → 建模 → 保存模型 → 检测 → 记录 → 导出 → 备份 → 恢复。"""
    det_id = _build_env(client)
    # 检测结果存在
    row = db.query_one(Path(client.application.config['DATA_DIR']) / 'fluro.db',
                       'SELECT * FROM detections WHERE image_id=?', (det_id,))
    assert row is not None and row['conc'] is not None
    # 首页汇总
    s = client.get('/api/home/summary').get_json()
    assert s['model_name'] == 'M9' and s['last_detection']['image_id'] == det_id
    # 记录检索
    rows = client.get('/api/detections?q=within').get_json()['detections']
    assert len(rows) == 1
    # 导出 CSV
    resp = client.get('/api/detections/export')
    assert resp.status_code == 200 and 'csv' in resp.mimetype
    # 备份 → 恢复
    b = client.post('/api/backup', json={}).get_json()
    assert b['ok']
    db.set_setting(Path(client.application.config['DATA_DIR']) / 'fluro.db', 'current_batch', 'changed')
    r = client.post('/api/backup/restore', json={'name': b['backup']['name']}).get_json()
    assert r['ok']
    assert db.get_setting(Path(client.application.config['DATA_DIR']) / 'fluro.db', 'current_batch') is None


def test_manual_fix_cascades(client):
    """手动修正 ROI → 特征与检测结果自动重算（D 验收）。"""
    det_id = _build_env(client)
    d1 = client.post(f'/api/detect/{det_id}', json={}).get_json()['detection']
    client.post(f'/api/pipeline/{det_id}/roi', json={'x': 0.0, 'y': 0.0, 'w': 0.1, 'h': 0.1, 'source': 'manual'})
    d2 = client.post(f'/api/detect/{det_id}', json={}).get_json()['detection']
    assert d1['conc'] != d2['conc']


# ---------- 性能基线 ----------

def test_pipeline_performance_single(client):
    """单张流水线（预处理+ROI+特征+检测）≤ 5s。"""
    det_id = _build_env(client)   # 环境已含一张检测图
    t0 = time.time()
    client.post(f'/api/pipeline/{det_id}/run', json={})
    dt = time.time() - t0
    assert dt < 5.0, f'单张流水线耗时 {dt:.2f}s 超过 5s'


def test_pipeline_performance_batch(client):
    """10 张批量流水线 ≤ 60s。"""
    # 用新 client 建环境（避免叠加）
    _upload(client, 50, kind='calibration')
    client.post('/api/templates', json={'name': 'T', 'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44})
    client.post('/api/models', json={
        'name': 'M', 'type': 'linear', 'params': {'a': -0.6, 'b': 60},
        'metrics': {'r2': 0.99, 'rmse': 0.5, 'lod': 1.0},
        'source_snapshot': {'feature': 'hue', 'data': [[0, 60], [25, 45], [50, 30], [75, 15], [100, 0]]},
    })
    ids = [_upload(client, 50, kind='detection', batch='B9P') for _ in range(10)]
    t0 = time.time()
    client.post('/api/pipeline/run', json={'image_ids': ids})
    dt = time.time() - t0
    assert dt < 60.0, f'10 张批量流水线耗时 {dt:.2f}s 超过 60s'


# ---------- 边界与友好错误 ----------

def test_empty_startup_state(client):
    """全新数据目录：首页为空状态、无检测记录、无模型。"""
    s = client.get('/api/home/summary').get_json()
    assert s['last_detection'] is None and s['model_name'] is None
    assert client.get('/api/detections').get_json()['detections'] == []
    assert client.get('/api/models').get_json()['models'] == []


def test_friendly_errors(client):
    """无模型检测、坏特征拟合均返回友好错误信息。"""
    iid = _upload(client, 50, kind='detection')
    client.post('/api/templates', json={'name': 'T', 'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44})
    client.post(f'/api/pipeline/{iid}/run', json={})
    resp = client.post(f'/api/detect/{iid}', json={})
    assert resp.status_code == 400
    assert '模型' in resp.get_json()['error']
    resp2 = client.post('/api/calibration/fit', json={'feature': 'hue'})
    assert resp2.status_code == 400
    assert '标定' in resp2.get_json()['error'] or '不足' in resp2.get_json()['error']


def test_bad_upload_rejected(client):
    """非图片文件上传应被拒绝且不影响后续。"""
    resp = client.post('/api/images/upload',
                       data={'files': [(io.BytesIO(b'not an image'), 'x.txt')], 'kind': 'detection', 'batch': 'B9'},
                       content_type='multipart/form-data')
    assert resp.status_code == 200
    j = resp.get_json()
    assert j['images'] == [] and len(j.get('errors', [])) >= 1
