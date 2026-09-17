# -*- coding: utf-8 -*-
"""问题1：总览页展示最近检测工作流的每步处理图像（原图/预处理/通道/ROI）。"""
import io

import pytest

from app import create_app  # noqa: E402
from tools import make_test_images as mti  # noqa: E402

ROIS_T_BG = [{'name': 'T', 'x': 0.325, 'y': 0.267, 'w': 0.35, 'h': 0.467},
             {'name': 'Bg', 'x': 0.1, 'y': 0.133, 'w': 0.25, 'h': 0.5}]


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    with app.test_client() as c:
        yield c


def _upload(client):
    import cv2
    img = mti.make_test_image(50)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    iid = client.post('/api/images/upload',
                      data={'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': 'detection'},
                      content_type='multipart/form-data').get_json()['images'][0]['id']
    return iid


def _make_calib_csv(rows):
    s = 'concentration,T_R_over_Bg_R\n' + ''.join(f'{a},{b}\n' for a, b in rows)
    return io.BytesIO(s.encode('utf-8'))


def _full_setup(client):
    """标定 → 拟合 → 保存模型；返回检测图 id（已检测）。"""
    iid = _upload(client)
    client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
    client.post(f'/api/images/{iid}/features', json={})
    rows = [(0, 1.0), (1, 0.91), (2, 0.82), (5, 0.55), (10, 0.1)]
    client.post('/api/modeling/import',
                data={'file': (_make_calib_csv(rows), 'cal.csv'),
                      'conc_col': 'concentration', 'feature_col': 'T_R_over_Bg_R'},
                content_type='multipart/form-data')
    fit = client.post('/api/calibration/fit', json={'feature': 'T_R_over_Bg_R'}).get_json()
    lin = fit['results']['linear']
    client.post('/api/models', json={
        'name': '首页模型', 'type': 'linear', 'params': lin['params'],
        'metrics': {'r2': lin['r2'], 'rmse': lin['rmse'], 'lod': lin.get('lod')},
        'source_snapshot': {'feature': 'T_R_over_Bg_R', 'data': fit['data'], 'n': fit['n'],
                            'preprocess': fit['preprocess']},
    })
    # 跑完整流水线（含检测）产生各步图像
    r = client.post('/api/pipeline/run', json={'image_ids': [iid]})
    assert r.get_json()['results'][0]['steps']['result'] in ('ok', 'attention')
    return iid


def test_home_summary_workflow_steps_images(client):
    """首页摘要包含最近检测图的每步图像 URL，且对应接口可访问。"""
    iid = _full_setup(client)
    s = client.get('/api/home/summary').get_json()
    assert s['workflow_steps'] is not None
    wf = s['workflow_steps']
    assert wf['image_id'] == iid
    assert wf['upload'] == f'/api/images/{iid}/original'
    assert wf['preprocess'] == f'/api/images/{iid}/processed'
    assert set(wf['channels'].keys()) >= {'r', 'g', 'b'}
    assert wf['roi'] == f'/api/images/{iid}/overlay'
    # 各步图像接口真实可访问（PNG）
    for url in [wf['upload'], wf['preprocess']] + list(wf['channels'].values()) + [wf['roi']]:
        resp = client.get(url)
        assert resp.status_code == 200, f'{url} 不可访问'
        assert resp.data[:4] == b'\x89PNG', f'{url} 不是 PNG'


def test_home_summary_workflow_steps_empty(client):
    """无检测记录时 workflow_steps 为 None。"""
    s = client.get('/api/home/summary').get_json()
    assert s['workflow_steps'] is None

