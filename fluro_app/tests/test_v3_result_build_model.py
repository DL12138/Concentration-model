# -*- coding: utf-8 -*-
"""问题3：工作流步骤6（结果页）「生成模型」按钮的后端链路。

按钮前端行为 = 标定数据 → /api/calibration/fit（拟合）→ /api/models（保存生效）→
重新检测当前图并展示新模型结果。本测试验证该链路各接口契约：
- 无标定数据时拟合返回 400（前端提示数据不足）；
- 有标定数据时一键拟合+保存后模型立即生效；
- 生成模型后新图可立即用新模型检测（结果页可展示新模型结果）。
"""
import io

import pytest

from app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    with app.test_client() as c:
        yield c


def _make_calib_csv(rows):
    s = 'concentration,T_R_over_Bg_R\n' + ''.join(f'{a},{b}\n' for a, b in rows)
    return io.BytesIO(s.encode('utf-8'))


def _upload(client):
    from tools import make_test_images as mti
    import cv2
    img = mti.make_test_image(50)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    return client.post('/api/images/upload',
                       data={'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': 'detection'},
                       content_type='multipart/form-data').get_json()['images'][0]['id']


def test_build_model_no_data_returns_400(client):
    """无标定数据：一键生成模型应失败（无数据或无有效特征），前端据此提示先录入标定数据。"""
    # 按钮先检查数据量：无标定数据时前端直接提示，不会发出拟合请求
    data = client.get('/api/calibration/data').get_json()
    n = sum(len(g.get('points') or []) for g in data['groups'])
    assert n == 0
    # 即使发出拟合请求也应 400（不支持的特征/数据不足均合法）
    resp = client.post('/api/calibration/fit', json={'feature': 'T_R_over_Bg_R'})
    assert resp.status_code == 400


def test_build_model_chain(client):
    """一键生成模型链路：导入标定 → 拟合 → 保存生效 → 新图可用新模型检测。"""
    rows = [(0, 1.0), (1, 0.91), (2, 0.82), (5, 0.55), (10, 0.1)]
    client.post('/api/modeling/import',
                data={'file': (_make_calib_csv(rows), 'cal.csv'),
                      'conc_col': 'concentration', 'feature_col': 'T_R_over_Bg_R'},
                content_type='multipart/form-data')
    # 1) 拟合（结果页「生成模型」按钮第一步）
    fit = client.post('/api/calibration/fit', json={'feature': 'T_R_over_Bg_R'})
    assert fit.status_code == 200
    body = fit.get_json()
    assert body['ok'] and body['best'] and body['results'][body['best']]
    best = body['results'][body['best']]
    # 2) 保存为生效模型（按钮第二步）
    saved = client.post('/api/models', json={
        'name': '工作流生成模型', 'type': body['best'], 'params': best['params'],
        'metrics': {'r2': best['r2'], 'rmse': best['rmse'], 'lod': best['lod']},
        'source_snapshot': {'feature': body['feature'], 'data': body['data'], 'n': body['n'],
                            'preprocess': body['preprocess']},
    })
    assert saved.status_code == 200
    mid = saved.get_json()['id']
    models = client.get('/api/models').get_json()['models']
    active = next(m for m in models if m['is_active'])
    assert active['id'] == mid
    # 3) 新图立即可用新模型检测（结果页重新检测展示新模型结果）
    iid = _upload(client)
    client.post('/api/templates', json={'name': '模板', 'template_json': {
        'T': {'x': 0.325, 'y': 0.267, 'w': 0.35, 'h': 0.467, 'role': 'sample'},
        'Bg': {'x': 0.1, 'y': 0.133, 'w': 0.25, 'h': 0.5, 'role': 'background'},
    }})
    client.post('/api/pipeline/run', json={'image_ids': [iid]})
    det = client.post(f'/api/detect/{iid}', json={}).get_json()
    assert det['ok'] is True
    assert det['detection']['model_name'] == '工作流生成模型'
    assert det['detection']['conc'] is not None
