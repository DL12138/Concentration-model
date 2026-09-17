# -*- coding: utf-8 -*-
"""问题4：MVP 验收（端到端）——按用户问题4 的 MVP 清单逐项验收。

覆盖：手动矩形 ROI(T/Bg) → 白平衡/背景校正 → RGB/Lab/OD/ΔE/比值特征 →
CSV 导出 → 线性/多项式/4PL 建模 → 交叉验证 → 模型保存与预测 →
检测 C±U 与超限判定 → 记录检索 → 备份与恢复。
"""
import io
import json
from pathlib import Path

import pytest

from app import create_app  # noqa: E402
from app import database as db  # noqa: E402
from tools import make_test_images as mti  # noqa: E402

ROIS_T_BG = [{'name': 'T', 'x': 0.325, 'y': 0.267, 'w': 0.35, 'h': 0.467},
             {'name': 'Bg', 'x': 0.1, 'y': 0.133, 'w': 0.25, 'h': 0.5}]


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    with app.test_client() as c:
        yield c


def _upload(client, filename='t.png', kind='detection', conc=None, batch='B1'):
    import cv2 as _cv2
    img = mti.make_test_image(50)
    ok, buf = _cv2.imencode('.png', img[:, :, ::-1])
    data = {'files': [(io.BytesIO(buf.tobytes()), filename)], 'kind': kind}
    if kind == 'calibration':
        data['known_conc'] = str(conc) if conc is not None else '50'
    data['batch'] = batch
    return client.post('/api/images/upload', data=data,
                       content_type='multipart/form-data').get_json()['images'][0]['id']


def _make_calib_csv(rows):
    s = 'concentration,T_R_over_Bg_R\n' + ''.join(f'{a},{b}\n' for a, b in rows)
    return io.BytesIO(s.encode('utf-8'))


def _full_setup(client):
    """搭建：检测图 + 标定分组（导入 CSV）→ 特征 → 拟合 → 保存 → 返回检测图 id。"""
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
        'name': 'MVP 线性模型', 'type': 'linear', 'params': lin['params'],
        'metrics': {'r2': lin['r2'], 'rmse': lin['rmse'], 'lod': lin.get('lod')},
        'source_snapshot': {'feature': 'T_R_over_Bg_R', 'data': fit['data'], 'n': fit['n'],
                            'preprocess': fit['preprocess']},
    })
    return iid


def test_accept_roi_and_features(client):
    """MVP 1：手动矩形 ROI（T/Bg）+ 特征（RGB/Lab/OD/ΔE/比值）。"""
    iid = _upload(client)
    r = client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
    assert r.status_code == 200
    feats = client.post(f'/api/images/{iid}/features', json={}).get_json()
    names = set((feats.get('rois') or {}).keys())
    assert names >= {'T', 'Bg'}
    fj = feats['rois']['T']
    for k in ('mean_r', 'mean_g', 'mean_b', 'hue', 'saturation', 'value', 'lab_l', 'lab_a', 'lab_b', 'od_r'):
        assert k in fj, f'缺少特征 {k}'
    comb = feats.get('combined', {})
    assert 'T_R_over_Bg_R' in comb
    assert 'deltaE_T_vs_Bg' in comb


def test_accept_export_csv_columns(client):
    """MVP 2：特征导出 CSV 包含规格 25 列中的核心列。"""
    iid = _upload(client, kind='calibration', conc=2, batch='Cal')
    client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
    client.post(f'/api/images/{iid}/features', json={})
    r = client.get('/api/export/features.csv')
    assert r.status_code == 200
    head = r.data.decode('utf-8-sig').splitlines()[0]
    for col in ('image_name', 'batch', 'concentration', 'T_R', 'T_G', 'T_B',
                'Bg_R', 'Bg_G', 'Bg_B', 'deltaE_T_vs_Bg'):
        assert col in head, f'缺少导出列 {col}'


def test_accept_model_save_and_detect(client):
    """MVP 3：线性模型保存为生效模型 → 检测图输出 C±U 与超限判定。"""
    iid = _full_setup(client)
    det = client.post(f'/api/detect/{iid}', json={}).get_json()
    assert det['ok'] is True
    d = det['detection']
    assert d['conc'] is not None and d['u'] is not None
    assert d['status'] in ('within', 'above', 'below', 'borderline')
    assert d['model_name'] == 'MVP 线性模型'
    # 记录可检索
    rec = client.get('/api/detections').get_json()['detections']
    assert any(r['image_id'] == iid for r in rec)


def test_accept_cv_and_poly_4pl(client):
    """MVP 4：交叉验证 + 多项式/4PL 模型均可拟合出指标。"""
    rows = [(0, 1.0), (1, 0.91), (2, 0.82), (5, 0.55), (10, 0.1)]
    client.post('/api/modeling/import',
                data={'file': (_make_calib_csv(rows), 'cal.csv'),
                      'conc_col': 'concentration', 'feature_col': 'T_R_over_Bg_R'},
                content_type='multipart/form-data')
    fit = client.post('/api/calibration/fit', json={'feature': 'T_R_over_Bg_R'}).get_json()
    for mt in ('linear', 'poly2', '4pl', 'log', 'pls'):
        assert 'error' not in fit['results'][mt]
        assert fit['results'][mt]['r2'] is not None
    cv = client.post('/api/modeling/cv', json={'feature': 'T_R_over_Bg_R', 'method': 'kfold'}).get_json()
    assert 'summary' in cv['results']['linear']
    assert cv['results']['linear']['summary']['r2'] is not None


def test_accept_backup_restore(client):
    """MVP 5：备份与恢复不丢数据。"""
    iid = _full_setup(client)
    det = client.post(f'/api/detect/{iid}', json={}).get_json()
    assert det['ok'] is True
    r = client.post('/api/backup').get_json()
    assert r['ok'] is True
    bname = r['backup']['name']
    before = client.get('/api/detections').get_json()['detections']
    # 删除一条数据后恢复
    db.execute(Path(client.application.config['DATA_DIR']) / 'fluro.db',
               'DELETE FROM detections')
    assert client.get('/api/detections').get_json()['detections'] == []
    r2 = client.post('/api/backup/restore', json={'name': bname})
    assert r2.status_code == 200
    after = client.get('/api/detections').get_json()['detections']
    assert len(after) == len(before)
    assert after[0]['conc'] == before[0]['conc']


def test_accept_workflow_auto_and_manual_fix(client):
    """MVP 6：上传后自动流水线可跑通；每步允许人工修正（ROI 可改、预处理参数可调）。"""
    iid = _upload(client)
    pipe = client.post('/api/pipeline/run', json={'image_ids': [iid]}).get_json()
    assert pipe.get('ok') is True
    # 人工修正 ROI（替换坐标）
    r = client.post(f'/api/images/{iid}/rois', json={'rois': [
        {'name': 'T', 'x': 0.31, 'y': 0.25, 'w': 0.38, 'h': 0.5},
        {'name': 'Bg', 'x': 0.09, 'y': 0.12, 'w': 0.28, 'h': 0.53}]})
    assert r.status_code == 200
    # 修正 ROI 后重新提取特征
    client.post(f'/api/images/{iid}/features', json={})
    # 预处理参数调整（暗场/平场/去噪/白平衡）
    p = client.post(f'/api/pipeline/{iid}/preprocess', json={
        'filter': 'gaussian', 'kernel': 3, 'use_dark': False, 'use_flat': False,
        'use_wb': True, 'wb_roi_name': 'Bg'})
    assert p.status_code == 200
    # 标定并保存生效模型，使检测可用
    rows = [(0, 1.0), (1, 0.91), (2, 0.82), (5, 0.55), (10, 0.1)]
    client.post('/api/modeling/import',
                data={'file': (_make_calib_csv(rows), 'cal.csv'),
                      'conc_col': 'concentration', 'feature_col': 'T_R_over_Bg_R'},
                content_type='multipart/form-data')
    fit = client.post('/api/calibration/fit', json={'feature': 'T_R_over_Bg_R'}).get_json()
    lin = fit['results']['linear']
    client.post('/api/models', json={
        'name': 'MVP 模型', 'type': 'linear', 'params': lin['params'], 'metrics': {},
        'source_snapshot': {'feature': 'T_R_over_Bg_R', 'data': fit['data'], 'n': fit['n'],
                            'preprocess': fit['preprocess']},
    })
    # 修正后重新检测仍可用
    det = client.post(f'/api/detect/{iid}', json={}).get_json()
    assert det['ok'] is True
