# -*- coding: utf-8 -*-
"""问题 2-K：模型文件导出/导入（joblib）与特征 CSV 批量预测。"""
import io
import json
from pathlib import Path

import pytest
import numpy as np  # noqa: E402

from app import create_app  # noqa: E402
from app import database as db  # noqa: E402
from tools import make_test_images as mti  # noqa: E402

ROIS_T_BG = [{'name': 'T', 'x': 260, 'y': 160, 'w': 280, 'h': 280},
             {'name': 'Bg', 'x': 80, 'y': 80, 'w': 200, 'h': 300}]


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    with app.test_client() as c:
        yield c


def _make_calib_csv(rows):
    s = 'concentration,T_R_over_Bg_R\n'
    for c, f in rows:
        s += f'{c},{f}\n'
    return io.BytesIO(s.encode('utf-8'))


def test_p2k_export_and_import_model_file(client):
    """模型导出为 .joblib、再导入复用（类型/参数一致）。"""
    rows = [(0, 1.0), (1, 0.91), (2, 0.82), (5, 0.55), (10, 0.1)]
    data = {'file': (_make_calib_csv(rows), 'cal.csv'),
            'conc_col': 'concentration', 'feature_col': 'T_R_over_Bg_R'}
    client.post('/api/modeling/import', data=data, content_type='multipart/form-data')
    fit = client.post('/api/calibration/fit', json={'feature': 'T_R_over_Bg_R'}).get_json()
    lin = fit['results']['linear']
    mid = client.post('/api/models', json={
        'name': '线性模型', 'type': 'linear', 'params': lin['params'],
        'metrics': {'r2': lin['r2']},
        'source_snapshot': {'feature': 'T_R_over_Bg_R', 'data': fit['data'], 'n': fit['n'],
                            'preprocess': fit['preprocess']},
    }).get_json()['id']
    r = client.get(f'/api/models/{mid}/export')
    assert r.status_code == 200
    import joblib
    payload = joblib.load(io.BytesIO(r.data))
    assert payload['type'] == 'linear'
    assert payload['params']['a'] == pytest.approx(lin['params']['a'], rel=0.01)
    r2 = client.post('/api/models/import_joblib',
                     data={'file': (io.BytesIO(r.data), 'm.joblib')},
                     content_type='multipart/form-data')
    assert r2.status_code == 200
    d = r2.get_json()
    assert d['type'] == 'linear'
    assert d['name'] == '线性模型'
    models = client.get('/api/models').get_json()['models']
    assert any(m['id'] == d['id'] and m['type'] == 'linear' for m in models)


def test_p2k_predict_csv_batch(client):
    """特征 CSV 批量预测：逐行输出 C±U 与超限判定。"""
    rows = [(0, 1.0), (1, 0.91), (2, 0.82), (5, 0.55), (10, 0.1)]
    data = {'file': (_make_calib_csv(rows), 'cal.csv'),
            'conc_col': 'concentration', 'feature_col': 'T_R_over_Bg_R'}
    client.post('/api/modeling/import', data=data, content_type='multipart/form-data')
    fit = client.post('/api/calibration/fit', json={'feature': 'T_R_over_Bg_R'}).get_json()
    lin = fit['results']['linear']
    mid = client.post('/api/models', json={
        'name': '线性模型', 'type': 'linear', 'params': lin['params'],
        'metrics': {'r2': lin['r2']},
        'source_snapshot': {'feature': 'T_R_over_Bg_R', 'data': fit['data'], 'n': fit['n'],
                            'preprocess': fit['preprocess']},
    }).get_json()['id']
    client.post(f'/api/models/{mid}/activate')
    client.post('/api/settings', json={'settings': {'limit_lower': 0, 'limit_upper': 5}})
    feat_csv = 'T_R_over_Bg_R\n1.0\n0.82\n0.2\n'
    r = client.post('/api/modeling/predict_csv',
                    data={'file': (io.BytesIO(feat_csv.encode('utf-8-sig')), 'f.csv')},
                    content_type='multipart/form-data')
    assert r.status_code == 200
    d = r.get_json()
    preds = d['predictions']
    assert len(preds) == 3
    # 特征 1.0 → 浓度 0；0.82 → 浓度 2
    assert preds[0]['conc'] == pytest.approx(0.0, abs=0.15)
    assert preds[1]['conc'] == pytest.approx(2.0, abs=0.2)
    assert 'u' in preds[1] and preds[1]['u'] >= 0
    assert preds[2]['status'] == 'above'  # 浓度 9.8 > 上限 5
    # 导出 CSV 直接下载
    r2 = client.post('/api/modeling/predict_csv/export',
                     data={'file': (io.BytesIO(feat_csv.encode('utf-8-sig')), 'f.csv')},
                     content_type='multipart/form-data')
    assert r2.status_code == 200
    text = r2.data.decode('utf-8-sig')
    assert 'predicted_conc' in text
    assert text.count('\n') == 4  # 表头 + 3 行


def test_p2k_predict_csv_missing_feature_col(client):
    """CSV 缺少模型特征列 → 400 并给出特征名提示。"""
    rows = [(0, 1.0), (1, 0.91), (2, 0.82), (5, 0.55), (10, 0.1)]
    data = {'file': (_make_calib_csv(rows), 'cal.csv'),
            'conc_col': 'concentration', 'feature_col': 'T_R_over_Bg_R'}
    client.post('/api/modeling/import', data=data, content_type='multipart/form-data')
    fit = client.post('/api/calibration/fit', json={'feature': 'T_R_over_Bg_R'}).get_json()
    lin = fit['results']['linear']
    client.post('/api/models', json={
        'name': '线性模型', 'type': 'linear', 'params': lin['params'], 'metrics': {},
        'source_snapshot': {'feature': 'T_R_over_Bg_R', 'data': fit['data'], 'n': fit['n'],
                            'preprocess': fit['preprocess']},
    })
    r = client.post('/api/modeling/predict_csv',
                    data={'file': (io.BytesIO(b'other,value\n1,2\n'), 'f.csv')},
                    content_type='multipart/form-data')
    assert r.status_code == 400
    assert 'T_R_over_Bg_R' in r.get_json()['error']
