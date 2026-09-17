# -*- coding: utf-8 -*-
"""第六批：批量删除功能（上传界面缩略图墙勾选多张 → 批量删除，
清理特征/ROI/流水线/检测记录/标定点关联记录与磁盘文件）。"""
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


def _upload(client, kind='calibration', conc='10'):
    files = [(_img_bytes(), 'f.png')]
    data = {'files': files, 'kind': kind}
    if kind == 'calibration':
        data['known_conc'] = conc
    return client.post('/api/images/upload', data=data,
                       content_type='multipart/form-data').get_json()['images'][0]


def test_batch_delete_images_and_cleanup(client):
    """批量删除两张标定图：记录/特征/ROI 清除，剩余图片保留，重复删除计数为跳过。"""
    a, b, cimg = _upload(client), _upload(client), _upload(client)
    ids = [a['id'], b['id'], cimg['id']]
    for iid in ids:
        client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
        client.post(f'/api/images/{iid}/features', json={})
    # 删除前：3 张
    assert len(client.get('/api/images').get_json()['images']) == 3
    r = client.post('/api/images/batch_delete', json={'ids': [a['id'], b['id']]})
    assert r.status_code == 200
    assert r.get_json() == {'ok': True, 'deleted': 2, 'skipped': 0}
    remaining = client.get('/api/images').get_json()['images']
    assert len(remaining) == 1
    assert remaining[0]['id'] == cimg['id']
    # 已删除图的特征接口 404（关联记录已清理）
    assert client.get(f"/api/images/{a['id']}/features").status_code == 404
    # 再删不存在的图 → 跳过
    r2 = client.post('/api/images/batch_delete', json={'ids': [a['id'], 9999]})
    assert r2.get_json() == {'ok': True, 'deleted': 0, 'skipped': 2}


def test_batch_delete_removes_detection_and_group(client):
    """批量删除已检测/已入标定分组的图：检测记录与标定分组点同步清除。"""
    # 三张标定图建线性模型（拟合需≥3 个浓度）
    cal5 = _upload(client, kind='calibration', conc='5')
    cal10 = _upload(client, kind='calibration', conc='10')
    cal20 = _upload(client, kind='calibration', conc='20')
    for im in (cal5, cal10, cal20):
        client.post(f'/api/images/{im["id"]}/rois', json={'rois': ROIS_T_BG})
        client.post(f'/api/images/{im["id"]}/features', json={})
        client.post('/api/calibration/quick', json={'image_id': im['id'], 'conc': float(im['known_conc'])})
    fit = client.post('/api/calibration/fit', json={'feature': 'hue'}).get_json()
    lin = fit['results']['linear']
    client.post('/api/models', json={'name': 'M', 'type': 'linear', 'params': lin['params'],
                                     'metrics': {}, 'source_snapshot': {'feature': 'hue', 'data': fit['data'],
                                                                        'n': fit['n'], 'unit': fit['unit'],
                                                                        'preprocess': fit['preprocess']}})
    # 检测图：检测成功并写入记录
    det = _upload(client, kind='detection')
    client.post(f'/api/images/{det["id"]}/rois', json={'rois': ROIS_T_BG})
    client.post(f'/api/images/{det["id"]}/features', json={})
    dr = client.post('/api/detect/' + str(det['id']), json={})
    assert dr.status_code == 200
    dets = client.get('/api/detections').get_json()['detections']
    assert any(d['image_id'] == det['id'] for d in dets)
    # 标定数据：三组各 1 点
    data = client.get('/api/calibration/data').get_json()
    assert len(data['groups']) == 3
    # 批量删除：一张已检测图 + 一张已入组标定图（cal5）
    r = client.post('/api/images/batch_delete',
                    json={'ids': [det['id'], cal5['id']]})
    assert r.get_json()['deleted'] == 2
    # 检测记录清除
    dets2 = client.get('/api/detections').get_json()['detections']
    assert all(d['image_id'] != det['id'] for d in dets2)
    # 标定分组点清除（cal5 组被删空，剩 conc10 与 conc20 两组）
    data2 = client.get('/api/calibration/data').get_json()
    assert len(data2['groups']) == 2
    assert all(g['conc'] != 5 for g in data2['groups'])
    assert all(p['image_id'] != cal5['id'] for g in data2['groups'] for p in g['points'])


def test_batch_delete_validation(client):
    """批量删除参数校验：空列表/非数字拒绝。"""
    assert client.post('/api/images/batch_delete', json={}).status_code == 400
    assert client.post('/api/images/batch_delete', json={'ids': []}).status_code == 400
    assert client.post('/api/images/batch_delete', json={'ids': ['x', 'y']}).status_code == 400


def test_single_delete_still_works(client):
    """单张删除接口保持可用（回归）。"""
    im = _upload(client)
    assert client.delete(f"/api/images/{im['id']}").get_json()['ok'] is True
    assert client.get('/api/images').get_json()['images'] == []
    assert client.delete(f"/api/images/{im['id']}").status_code == 404
