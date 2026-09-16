# -*- coding: utf-8 -*-
"""M5 标定建模测试：分组管理、数据点纳入/剔除、4 模型拟合指标、模型库、端到端标定。"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import cv2  # noqa: E402

from app import create_app  # noqa: E402
from app import database as db  # noqa: E402
from app import modeling  # noqa: E402
from tools import make_test_images as mti  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    return app.test_client()


def _upload(client, conc, reps=1, kind='calibration'):
    ids = []
    for _ in range(reps):
        img = mti.make_test_image(conc)
        ok, buf = cv2.imencode('.png', img[:, :, ::-1])
        assert ok
        data = {'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': kind, 'batch': 'B1',
                'known_conc': str(conc)}
        resp = client.post('/api/images/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 200
        ids.append(resp.get_json()['images'][0]['id'])
    return ids


def _setup_calibration(client, concs=(0, 25, 50, 75, 100), reps=1):
    """上传标定图 + 建模板（紧贴 800x600 检测圆）+ 跑流水线 + 建分组 + 加入数据点。"""
    ids = []
    for c in concs:
        ids.extend(_upload(client, c, reps=reps))
    # 800x600 图中检测圆：圆心 (400,300)，r=132 → x∈[0.335,0.665]，y∈[0.28,0.72]
    client.post('/api/templates', json={'name': 'T', 'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44})
    client.post('/api/pipeline/run', json={})
    gids = {}
    for c in concs:
        gres = client.post('/api/calibration/groups', json={'conc': c}).get_json()
        gids[c] = gres['id']
    # 把每个图加入对应浓度分组
    for i, c in enumerate(concs):
        for r in range(reps):
            pid = client.post('/api/calibration/points',
                              json={'image_id': ids[i * reps + r], 'group_id': gids[c]}).get_json()
            assert pid['ok']
    return ids


def test_group_create_reuse_delete(client):
    r1 = client.post('/api/calibration/groups', json={'conc': 10, 'name': 'A'}).get_json()
    assert r1['reused'] is False
    r2 = client.post('/api/calibration/groups', json={'conc': 10}).get_json()
    assert r2['reused'] is True and r2['id'] == r1['id']
    assert client.post('/api/calibration/groups', json={'conc': 'abc'}).status_code == 400
    resp = client.delete(f"/api/calibration/groups/{r1['id']}")
    assert resp.get_json()['ok']


def test_add_point_requires_features(client):
    iid = _upload(client, 50)[0]
    gres = client.post('/api/calibration/groups', json={'conc': 50}).get_json()
    # 未跑流水线（无特征）
    resp = client.post('/api/calibration/points', json={'image_id': iid, 'group_id': gres['id']})
    assert resp.status_code == 400


def test_fit_requires_min_points(client):
    resp = client.post('/api/calibration/fit', json={'feature': 'hue'})
    assert resp.status_code == 400
    _setup_calibration(client, concs=(10, 20))  # 只有 2 个浓度
    resp = client.post('/api/calibration/fit', json={'feature': 'hue'})
    assert resp.status_code == 400


def test_fit_returns_four_models_and_best(client):
    _setup_calibration(client)
    resp = client.post('/api/calibration/fit', json={'feature': 'ratio_gr'})
    assert resp.status_code == 200
    j = resp.get_json()
    assert j['ok'] and j['n'] == 5
    assert set(j['results'].keys()) == {'linear', 'poly2', 'exp', '4pl'}
    # G/R 比值随浓度单调下降：conc=0 → ~5.8，conc=100 → ~0.17
    lin = j['results']['linear']
    assert 'error' not in lin
    assert -0.09 < lin['params']['a'] < -0.03
    assert 3.5 < lin['params']['b'] < 7
    assert lin['r2'] > 0.7
    assert lin['rmse'] < 2
    # 数据带明显曲率，二次多项式应优于线性
    p2 = j['results']['poly2']
    assert 'error' not in p2
    assert p2['r2'] > lin['r2']
    assert j['best'] in ('linear', 'poly2', 'exp', '4pl')


def test_fit_three_points_4pl_reports_error_not_crash(client):
    """只有 3 个浓度点时：4PL（4 参数）应报告数据不足，接口返回 200 而非 500。"""
    _setup_calibration(client, concs=(0, 50, 100))
    resp = client.post('/api/calibration/fit', json={'feature': 'ratio_gr'})
    assert resp.status_code == 200
    j = resp.get_json()
    assert '4pl' in j['results'] and 'error' in j['results']['4pl']
    assert '数据点' in j['results']['4pl']['error']
    # 其他低参数模型仍正常
    assert 'error' not in j['results']['linear']
    assert j['best'] in ('linear', 'poly2', 'exp')


def test_toggle_excludes_point(client):
    _setup_calibration(client)
    # 全部纳入时 n=5
    assert client.post('/api/calibration/fit', json={'feature': 'ratio_gr'}).get_json()['n'] == 5
    # 剔除一个点
    data = client.get('/api/calibration/data').get_json()
    pid = data['groups'][0]['points'][0]['point_id']
    client.post(f'/api/calibration/points/{pid}', json={'included': 0})
    assert client.post('/api/calibration/fit', json={'feature': 'ratio_gr'}).get_json()['n'] == 4
    # 恢复
    client.post(f'/api/calibration/points/{pid}', json={'included': 1})
    assert client.post('/api/calibration/fit', json={'feature': 'ratio_gr'}).get_json()['n'] == 5


def test_lod_with_replicates(client):
    """最低浓度 2 个重复点时应能算出 LOD（3.3σ/斜率）。"""
    _setup_calibration(client, reps=2)
    j = client.post('/api/calibration/fit', json={'feature': 'ratio_gr'}).get_json()
    lod = j['results']['linear']['lod']
    assert lod is not None and lod > 0


def test_model_save_activate_delete(client):
    _setup_calibration(client)
    j = client.post('/api/calibration/fit', json={'feature': 'ratio_gr'}).get_json()
    lin = j['results']['linear']
    r1 = client.post('/api/models', json={
        'name': 'M1', 'type': 'linear', 'params': lin['params'],
        'metrics': {'r2': lin['r2'], 'rmse': lin['rmse'], 'lod': lin['lod']},
    }).get_json()
    assert r1['ok']
    # 首个模型自动激活
    models = client.get('/api/models').get_json()['models']
    assert models[0]['is_active'] == 1
    # 保存第二个并切换激活
    r2 = client.post('/api/models', json={
        'name': 'M2', 'type': 'poly2',
        'params': j['results']['poly2']['params'], 'metrics': {},
    }).get_json()
    client.post(f"/api/models/{r2['id']}/activate", json={})
    models = client.get('/api/models').get_json()['models']
    active = [m for m in models if m['is_active']]
    assert len(active) == 1 and active[0]['id'] == r2['id']
    # 删除激活模型 → 剩余模型自动激活
    client.delete(f"/api/models/{r2['id']}")
    models = client.get('/api/models').get_json()['models']
    active = [m for m in models if m['is_active']]
    assert len(active) == 1 and active[0]['id'] == r1['id']


def test_end_to_end_predict(client):
    """端到端：标定 → 保存生效模型 → 预测已知特征得到接近的浓度。"""
    _setup_calibration(client)
    j = client.post('/api/calibration/fit', json={'feature': 'ratio_gr'}).get_json()
    lin = j['results']['linear']
    client.post('/api/models', json={'name': 'M', 'type': 'linear', 'params': lin['params'],
                                     'metrics': {'r2': lin['r2'], 'rmse': lin['rmse'], 'lod': lin['lod']}})
    # 已知 conc=50 时 G/R ≈ 1.0
    p = modeling.predict_conc('linear', lin['params'], 1.0)
    assert abs(p - 50) < 20
    pu = modeling.predict_with_u('linear', lin['params'], 1.0,
                                 [0, 25, 50, 75, 100], [5.78, 1.70, 1.00, 0.59, 0.17])
    assert abs(pu['conc'] - 50) < 20
    assert pu['u'] > 0


def test_predict_4pl_and_bounds(client):
    """4PL 反解与越界保护。"""
    _setup_calibration(client)
    j = client.post('/api/calibration/fit', json={'feature': 'ratio_gr'}).get_json()
    f4 = j['results']['4pl']
    assert 'error' not in f4
    # 曲线中点处的特征 → 反解应接近中点浓度 c
    p = f4['params']
    mid_feat = (p['a'] + p['d']) / 2.0
    c = modeling.predict_conc('4pl', f4['params'], mid_feat)
    assert abs(c - p['c']) / max(p['c'], 1e-6) < 0.5
    with pytest.raises(ValueError):
        modeling.predict_conc('4pl', f4['params'], p['a'] + 1e6)  # 超出范围
