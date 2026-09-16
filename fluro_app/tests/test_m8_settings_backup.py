# -*- coding: utf-8 -*-
"""M8 设置与备份恢复测试：设置读写、备份创建/列表/恢复/删除。"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import cv2  # noqa: E402

from app import create_app  # noqa: E402
from app import database as db  # noqa: E402
from app import backup as backup_mod  # noqa: E402
from tools import make_test_images as mti  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    return app.test_client()


def _upload(client, conc=50, kind='detection', batch='B8'):
    img = mti.make_test_image(conc)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    assert ok
    data = {'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': kind, 'batch': batch,
            'known_conc': str(conc)}
    resp = client.post('/api/images/upload', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    return resp.get_json()['images'][0]['id']


def test_settings_save_and_get(client):
    r = client.post('/api/settings', json={'settings': {
        'limit_lower': '0', 'limit_upper': '100', 'unit': 'ng/mL', 'current_batch': 'B8-1',
    }})
    assert r.get_json()['ok']
    s = client.get('/api/settings').get_json()['settings']
    assert s['limit_lower'] == '0'
    assert s['limit_upper'] == '100'
    assert s['current_batch'] == 'B8-1'
    # 清空限值
    client.post('/api/settings', json={'settings': {'limit_lower': None}})
    s = client.get('/api/settings').get_json()['settings']
    assert s['limit_lower'] is None


def test_settings_affects_detection(client):
    """设置上下限后检测判定应使用新限值（端到端）。"""
    cal_id = _upload(client, 50, kind='calibration')
    client.post('/api/templates', json={'name': 'T', 'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44})
    g = client.post('/api/calibration/groups', json={'conc': 50}).get_json()
    client.post('/api/calibration/points', json={'image_id': cal_id, 'group_id': g['id']})
    client.post('/api/models', json={
        'name': 'M', 'type': 'linear', 'params': {'a': -0.6, 'b': 60},
        'metrics': {'r2': 0.999, 'rmse': 0.5, 'lod': 1.0},
        'source_snapshot': {'feature': 'hue', 'data': [[0, 61], [25, 44], [50, 31], [75, 14], [100, 1]]},
    })
    iid = _upload(client, batch='B8X')
    client.post(f'/api/pipeline/{iid}/run', json={})
    client.post('/api/settings', json={'settings': {'limit_lower': '0', 'limit_upper': '40'}})
    d = client.post(f'/api/detect/{iid}', json={}).get_json()['detection']
    assert d['status'] in ('above', 'borderline')


def test_backup_create_list_restore(client, tmp_path):
    data_dir = Path(tmp_path)
    # 造一些数据
    iid = _upload(client, 50, kind='detection', batch='B8')
    client.post('/api/settings', json={'settings': {'limit_upper': '80'}})
    # 备份
    b = backup_mod.create_backup(data_dir)
    assert b['ok'] and (data_dir.parent / 'backups' / b['name'] / 'fluro.db').exists()
    lst = backup_mod.list_backups(data_dir)
    assert any(x['name'] == b['name'] for x in lst)
    # 修改数据后恢复
    db.set_setting(data_dir / 'fluro.db', 'limit_upper', '999')
    backup_mod.restore_backup(data_dir, b['name'])
    assert db.get_setting(data_dir / 'fluro.db', 'limit_upper') == '80'
    # 图片记录仍在
    rows = db.query(data_dir / 'fluro.db', 'SELECT id FROM images')
    assert any(r['id'] == iid for r in rows)
    # 恢复会产生安全备份
    lst2 = backup_mod.list_backups(data_dir)
    assert len(lst2) >= 2


def test_backup_api_endpoints(client):
    _upload(client)
    r = client.post('/api/backup', json={}).get_json()
    assert r['ok']
    name = r['backup']['name']
    lst = client.get('/api/backups').get_json()['backups']
    assert any(x['name'] == name for x in lst)
    # 删除备份
    d = client.post('/api/backup/' + name + '/delete', json={})
    assert d.get_json()['ok']
    lst2 = client.get('/api/backups').get_json()['backups']
    assert all(x['name'] != name for x in lst2)


def test_restore_missing_backup(client):
    resp = client.post('/api/backup/restore', json={'name': 'nope'})
    assert resp.status_code == 400


def test_settings_page_data(client):
    """设置页聚合数据：模板/模型/备份/数据目录齐全。"""
    _upload(client, 50, kind='calibration')
    client.post('/api/templates', json={'name': 'T', 'x': 0.1, 'y': 0.1, 'w': 0.3, 'h': 0.3})
    client.post('/api/models', json={'name': 'M', 'type': 'linear', 'params': {'a': 1, 'b': 1}})
    s = client.get('/api/settings').get_json()
    assert len(s['templates']) == 1
    assert len(s['models']) == 1
    assert s['data_dir']
    assert isinstance(s['backups'], list)
