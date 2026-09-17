# -*- coding: utf-8 -*-
"""第五批问题1：标定重复测量（一次上传 3 张同浓度标定图，自动编号重复 1..3，
同时处理，并在标定建模页输出每张图结果与均值±SD/CV 误差）。"""
import io

import pytest

from app import create_app  # noqa: E402

ROIS_T_BG = [{'name': 'T', 'x': 0.325, 'y': 0.267, 'w': 0.35, 'h': 0.467},
             {'name': 'Bg', 'x': 0.1, 'y': 0.133, 'w': 0.25, 'h': 0.5}]


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    with app.test_client() as c:
        yield c


def _img_bytes():
    from tools import make_test_images as mti
    import cv2
    img = mti.make_test_image(50)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    return io.BytesIO(buf.tobytes())


def test_upload_calibration_batch_replicates(client):
    """一次上传 3 张标定图：自动编号重复 1、2、3；检测图默认重复 1。"""
    files = [(_img_bytes(), 'r1.png'), (_img_bytes(), 'r2.png'), (_img_bytes(), 'r3.png')]
    r = client.post('/api/images/upload',
                    data={'files': files, 'kind': 'calibration', 'known_conc': '5',
                          'conc_unit': 'μg/mL'},
                    content_type='multipart/form-data').get_json()
    assert len(r['images']) == 3
    reps = [im['replicate'] for im in sorted(r['images'], key=lambda x: x['id'])]
    assert reps == [1, 2, 3]
    # 列表接口可见
    imgs = client.get('/api/images').get_json()['images']
    assert sorted(i['replicate'] for i in imgs if i['kind'] == 'calibration') == [1, 2, 3]
    # 检测图默认 replicate=1
    d = client.post('/api/images/upload',
                    data={'files': [(_img_bytes(), 'd.png')], 'kind': 'detection'},
                    content_type='multipart/form-data').get_json()['images'][0]
    assert d['replicate'] == 1


def test_replicates_group_stats_with_three_images(client):
    """一次上传 3 张同浓度标定图（自动编号 1..3），处理并加入分组后：
    组内 3 个重复点（带编号）、均值/SD/CV 正确输出。"""
    files = [(_img_bytes(), 'r1.png'), (_img_bytes(), 'r2.png'), (_img_bytes(), 'r3.png')]
    r = client.post('/api/images/upload',
                    data={'files': files, 'kind': 'calibration', 'known_conc': '10'},
                    content_type='multipart/form-data').get_json()
    iids = sorted(im['id'] for im in r['images'])
    assert sorted(im['replicate'] for im in r['images']) == [1, 2, 3]
    for iid in iids:
        client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
        client.post(f'/api/images/{iid}/features', json={})
        client.post('/api/calibration/quick', json={'image_id': iid, 'conc': 10})
    data = client.get('/api/calibration/data').get_json()
    assert len(data['groups']) == 1
    g = data['groups'][0]
    assert g['n'] == 3
    # 每张重复点带 replicate 编号 1..3；均值/SD/CV 按三张图的特征值正确计算
    reps = sorted(p['replicate'] for p in g['points'])
    assert reps == [1, 2, 3]
    feat = 'hue'
    vals = [p['features'][feat] for p in g['points']]
    mean = round(sum(vals) / len(vals), 4)
    sd = round((sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5, 4)
    assert g['mean'][feat] == mean
    assert g['sd'][feat] == sd
    assert g['cv'][feat] == round(sd / mean * 100, 2)


def test_group_stats_cv_from_import(client):
    """CSV 导入同浓度 3 个不同重复点：均值/SD/CV% 计算正确（模拟真实重复误差）。"""
    rows = [(10, 0.90), (10, 0.95), (10, 0.88)]  # 同浓度 3 重复，特征略有差异
    s = 'concentration,T_R_over_Bg_R\n' + ''.join(f'{a},{b}\n' for a, b in rows)
    client.post('/api/modeling/import',
                data={'file': (io.BytesIO(s.encode('utf-8')), 'cal.csv'),
                      'conc_col': 'concentration', 'feature_col': 'T_R_over_Bg_R'},
                content_type='multipart/form-data')
    data = client.get('/api/calibration/data').get_json()
    assert len(data['groups']) == 1
    g = data['groups'][0]
    assert g['n'] == 3
    m = g['mean']['T_R_over_Bg_R']
    sd = g['sd']['T_R_over_Bg_R']
    cv = g['cv']['T_R_over_Bg_R']
    vals = [0.90, 0.95, 0.88]
    mean = round(sum(vals) / 3, 4)
    sd_exp = round((sum((v - mean) ** 2 for v in vals) / 2) ** 0.5, 4)
    assert m == mean
    assert sd == sd_exp
    assert cv == round(sd / mean * 100, 2)


def test_replicates_in_export_csv(client):
    """特征导出 CSV 含 replicate 列，一次上传的三张标定图对应 1、2、3。"""
    files = [(_img_bytes(), 'r1.png'), (_img_bytes(), 'r2.png'), (_img_bytes(), 'r3.png')]
    r = client.post('/api/images/upload',
                    data={'files': files, 'kind': 'calibration', 'known_conc': '10'},
                    content_type='multipart/form-data').get_json()
    for im in r['images']:
        client.post(f"/api/images/{im['id']}/rois", json={'rois': ROIS_T_BG})
        client.post(f"/api/images/{im['id']}/features", json={})
    r = client.get('/api/export/features.csv')
    assert r.status_code == 200
    text = r.data.decode('utf-8-sig')
    lines = text.strip().splitlines()
    header = lines[0].split(',')
    assert 'replicate' in header
    idx = header.index('replicate')
    reps = sorted(int(line.split(',')[idx]) for line in lines[1:])
    assert reps == [1, 2, 3]
