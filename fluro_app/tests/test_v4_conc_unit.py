# -*- coding: utf-8 -*-
"""第四批问题2：浓度单位选项（ng/mL、μg/mL、mg/mL、μmol/L、mmol/L、mol/L）。

覆盖上传 → 标定分组 → 拟合 → 模型快照 → 检测结果/首页 的完整单位链路，
以及非法单位拒绝、默认单位 ng/mL、PATCH 修改单位。
"""
import io

import pytest

from app import create_app  # noqa: E402
from app.config import Config  # noqa: E402

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


def _upload_calib(client, conc=10, unit='μmol/L'):
    return client.post('/api/images/upload',
                       data={'files': [(_img_bytes(), 'cal.png')], 'kind': 'calibration',
                             'known_conc': str(conc), 'conc_unit': unit},
                       content_type='multipart/form-data').get_json()['images'][0]['id']


def _make_calib_csv(rows):
    s = 'concentration,T_R_over_Bg_R\n' + ''.join(f'{a},{b}\n' for a, b in rows)
    return io.BytesIO(s.encode('utf-8'))


def test_upload_calibration_with_unit(client):
    """上传标定图带单位保存；缺省为 ng/mL；非法单位被拒绝。"""
    iid = _upload_calib(client, conc=10, unit='μmol/L')
    imgs = client.get('/api/images').get_json()['images']
    im = next(i for i in imgs if i['id'] == iid)
    assert im['conc_unit'] == 'μmol/L'
    # 不传单位默认 ng/mL
    iid2 = client.post('/api/images/upload',
                       data={'files': [(_img_bytes(), 'd.png')], 'kind': 'detection'},
                       content_type='multipart/form-data').get_json()['images'][0]['id']
    im2 = next(i for i in client.get('/api/images').get_json()['images'] if i['id'] == iid2)
    assert im2['conc_unit'] == 'ng/mL'
    # 非法单位 400
    r = client.post('/api/images/upload',
                    data={'files': [(_img_bytes(), 'x.png')], 'kind': 'calibration',
                          'known_conc': '1', 'conc_unit': 'kg'},
                    content_type='multipart/form-data')
    assert r.status_code == 400
    assert '单位' in r.get_json()['error']


def test_conc_unit_through_calibration_chain(client):
    """标定图单位贯穿：加入分组 → 分组单位 → 拟合 unit → 检测结果/首页显示。"""
    unit = 'μmol/L'
    # 5 个不同浓度标定图（不同浓度、单位 μmol/L）
    iids = []
    for conc in (0, 1, 2, 5, 10):
        iid = _upload_calib(client, conc=conc, unit=unit)
        iids.append(iid)
        client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
        client.post(f'/api/images/{iid}/features', json={})
        # 用 hue 作为特征（fit 默认）——此处用 quick 加入分组
        client.post('/api/calibration/quick', json={'image_id': iid, 'conc': conc})
    # 分组带单位
    data = client.get('/api/calibration/data').get_json()
    assert len(data['groups']) == 5
    assert all(g['unit'] == unit for g in data['groups'])
    # 拟合返回单位
    fit = client.post('/api/calibration/fit', json={'feature': 'hue'}).get_json()
    assert fit['unit'] == unit
    assert fit['best'] in ('linear', 'poly2', 'log', 'exp', '4pl', 'pls')  # best 恒为主模型
    # 保存模型（快照带 unit）：用 linear 确保反解在范围内
    best = fit['results']['linear']
    client.post('/api/models', json={
        'name': '单位模型', 'type': 'linear', 'params': best['params'],
        'metrics': {'r2': best['r2']},
        'source_snapshot': {'feature': fit['feature'], 'data': fit['data'], 'n': fit['n'],
                            'unit': fit['unit'], 'preprocess': fit['preprocess']},
    })
    # 检测图 → 检测结果带 unit
    det_img = client.post('/api/images/upload',
                          data={'files': [(_img_bytes(), 'det.png')], 'kind': 'detection'},
                          content_type='multipart/form-data').get_json()['images'][0]['id']
    client.post('/api/templates', json={'name': 't', 'template_json': {
        'T': {'x': 0.325, 'y': 0.267, 'w': 0.35, 'h': 0.467, 'role': 'sample'},
        'Bg': {'x': 0.1, 'y': 0.133, 'w': 0.25, 'h': 0.5, 'role': 'background'},
    }})
    client.post('/api/pipeline/run', json={'image_ids': [det_img]})
    det = client.post(f'/api/detect/{det_img}', json={}).get_json()
    assert det['ok'] is True
    assert det['detection']['unit'] == unit
    # 首页最近检测带 unit
    s = client.get('/api/home/summary').get_json()
    assert s['last_detection']['unit'] == unit


def test_import_csv_with_unit(client):
    """CSV 导入标定数据时可指定单位，拟合结果带该单位。"""
    rows = [(0, 1.0), (1, 0.91), (2, 0.82), (5, 0.55), (10, 0.1)]
    client.post('/api/modeling/import',
                data={'file': (_make_calib_csv(rows), 'cal.csv'),
                      'conc_col': 'concentration', 'feature_col': 'T_R_over_Bg_R',
                      'unit': 'mmol/L'},
                content_type='multipart/form-data')
    data = client.get('/api/calibration/data').get_json()
    assert data['groups'] and all(g['unit'] == 'mmol/L' for g in data['groups'])
    fit = client.post('/api/calibration/fit', json={'feature': 'T_R_over_Bg_R'}).get_json()
    assert fit['unit'] == 'mmol/L'


def test_patch_image_conc_unit(client):
    """PATCH 可修改图片浓度单位；非法单位拒绝。"""
    iid = _upload_calib(client, conc=1, unit='ng/mL')
    r = client.patch(f'/api/images/{iid}', json={'conc_unit': 'mg/mL'})
    assert r.status_code == 200
    im = next(i for i in client.get('/api/images').get_json()['images'] if i['id'] == iid)
    assert im['conc_unit'] == 'mg/mL'
    r2 = client.patch(f'/api/images/{iid}', json={'conc_unit': 'bad'})
    assert r2.status_code == 400


def test_config_units_include_requested(client):
    """需求列出的 6 个单位均可用。"""
    for u in ('μmol/L', 'mmol/L', 'mol/L', 'ng/mL', 'μg/mL', 'mg/mL'):
        assert u in Config.CONC_UNITS
