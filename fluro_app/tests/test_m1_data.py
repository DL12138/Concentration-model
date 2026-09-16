# -*- coding: utf-8 -*-
"""M1 数据层与基础页面测试：建表、CRUD、上传接口、合成图生成器、单页骨架。"""
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import create_app  # noqa: E402
from app import database as db  # noqa: E402

ALL_TABLES = [
    'images', 'pipeline_steps', 'features', 'roi', 'roi_templates',
    'calibration_groups', 'calibration_points', 'models', 'detections',
    'settings', 'batches', 'schema_version',
]


@pytest.fixture()
def app(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def test_init_db_creates_all_tables(tmp_path):
    db_path = tmp_path / 'fluro.db'
    db.init_db(db_path)
    conn = db.get_conn(db_path)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {r['name'] for r in rows}
    finally:
        conn.close()
    for t in ALL_TABLES:
        assert t in names, f'缺少表 {t}'


def test_init_db_idempotent(tmp_path):
    db_path = tmp_path / 'fluro.db'
    db.init_db(db_path)
    db.init_db(db_path)  # 二次执行不报错
    db.init_db(db_path)


def test_crud_execute_and_query(tmp_path):
    db_path = tmp_path / 'fluro.db'
    db.init_db(db_path)
    iid = db.execute(db_path, 'INSERT INTO images (file_path) VALUES (?)', ('a.png',))
    assert iid > 0
    row = db.query_one(db_path, 'SELECT * FROM images WHERE id=?', (iid,))
    assert row['file_path'] == 'a.png'
    assert row['kind'] == 'detection'  # 默认值
    assert row['status'] == 'uploaded'


def test_batch_insert(tmp_path):
    db_path = tmp_path / 'fluro.db'
    db.init_db(db_path)
    db.execute_many(db_path, 'INSERT INTO images (file_path, kind) VALUES (?,?)',
                    [('a.png', 'calibration'), ('b.png', 'detection')])
    rows = db.query(db_path, 'SELECT * FROM images ORDER BY id')
    assert len(rows) == 2


def test_upload_batch_images(client, tmp_path):
    files = []
    for i in range(3):
        img = np.full((100, 120, 3), 30 + i * 20, dtype=np.uint8)
        ok, buf = cv2.imencode('.png', img)
        assert ok
        files.append((io.BytesIO(buf.tobytes()), f'test_{i}.png'))
    resp = client.post('/api/images/upload', data={
        'files': files,
        'kind': 'detection',
        'batch': 'B001',
    }, content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert len(body['images']) == 3
    for im in body['images']:
        assert im['batch'] == 'B001'
        assert im['thumb_url']
        assert Path(im['thumb_url'].split('/')[-1])  # id 存在

    rows = db.query(Path(tmp_path) / 'fluro.db', 'SELECT * FROM images ORDER BY id')
    assert len(rows) == 3
    for r in rows:
        assert os.path.exists(r['file_path']), '原图文件不存在'
        assert os.path.exists(r['thumb_path']), '缩略图文件不存在'


def test_upload_calibration_requires_conc(client):
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', img)
    assert ok
    resp = client.post('/api/images/upload', data={
        'files': [(io.BytesIO(buf.tobytes()), 'c.png')],
        'kind': 'calibration',
    }, content_type='multipart/form-data')
    assert resp.status_code == 400
    assert '浓度' in resp.get_json()['error']


def test_upload_bad_file_rejected(client):
    resp = client.post('/api/images/upload', data={
        'files': [(io.BytesIO(b'not an image'), 'bad.png')],
        'kind': 'detection',
    }, content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['images'] == []
    assert body['errors'], '坏文件应记录错误'


def test_thumb_served(client, tmp_path):
    img = np.full((200, 300, 3), 80, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', img)
    assert ok
    resp = client.post('/api/images/upload', data={
        'files': [(io.BytesIO(buf.tobytes()), 't.png')],
        'kind': 'detection',
    }, content_type='multipart/form-data')
    iid = resp.get_json()['images'][0]['id']
    r = client.get(f'/api/images/{iid}/thumb')
    assert r.status_code == 200
    assert r.mimetype.startswith('image/')


def test_list_images(client, tmp_path):
    img = np.full((80, 80, 3), 60, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', img)
    assert ok
    for _ in range(2):
        client.post('/api/images/upload', data={
            'files': [(io.BytesIO(buf.tobytes()), 'x.png')],
            'kind': 'detection',
        }, content_type='multipart/form-data')
    r = client.get('/api/images')
    assert r.status_code == 200
    assert len(r.get_json()['images']) == 2


def test_make_test_image_color_monotonic():
    from tools import make_test_images as mti
    imgs = []
    for conc in [0, 25, 50, 75, 100]:
        img = mti.make_test_image(conc)
        assert img.shape == (600, 800, 3)
        imgs.append(img)
    # 检测区中心颜色：浓度增大，色相应从绿(≈120°)单调降至红(≈0°)
    hues = []
    for img in imgs:
        px = img[300, 400].astype(np.float64) / 255.0
        rgb = np.array([[[px[0], px[1], px[2]]]], dtype=np.float32)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0][0]
        hues.append(float(hsv[0]))
    # 处理色相环绕（0°=360°）：把每个值映射到与前一个最接近的等价角度，再断言单调递减
    norm = []
    prev = hues[0]
    for h in hues:
        while h - prev > 180:
            h -= 360
        while h - prev < -180:
            h += 360
        norm.append(h)
        prev = h
    assert all(norm[i] >= norm[i + 1] - 1 for i in range(len(norm) - 1)), f'色相应单调递减：{hues}'
    assert norm[0] > 90 and norm[-1] < 30


def test_make_batch_files(tmp_path):
    from tools import make_test_images as mti
    files = mti.make_batch([0, 50, 100], tmp_path, reps=2)
    assert len(files) == 6
    for f in files:
        assert os.path.exists(f)


def test_index_has_five_navs():
    idx = (ROOT / 'static' / 'index.html').read_text(encoding='utf-8')
    for view in ['home', 'workflow', 'modeling', 'records', 'settings']:
        assert f'data-view="{view}"' in idx
