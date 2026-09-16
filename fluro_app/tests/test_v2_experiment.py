# -*- coding: utf-8 -*-
"""v2 功能改进测试集：单卡片槽比色检测规格。

问题 1：实验信息配置（实验参数持久化 + 校验，默认 T/Bg 双 ROI）。
后续问题 2~4 在此文件逐步追加。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app import create_app  # noqa: E402
from app import database as db  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app({'TESTING': True, 'DATA_DIR': str(tmp_path)})
    return app.test_client()


# ============ 问题 1：实验信息配置 ============

def test_p1_default_experiment(client):
    ex = client.get('/api/experiment').get_json()
    assert ex['carrier'] == 'single_card'
    assert ex['roi_count'] == 2
    assert ex['roi_names'] == ['T', 'Bg']
    assert ex['has_blank_ref'] is True
    assert 'analyte' in ex and 'color_trend' in ex


def test_p1_save_and_load(client):
    payload = {
        'analyte': '某农药',
        'color_trend': '浓度越高 T 区越深',
        'roi_count': 2,
        'roi_names': ['T', 'Bg'],
        'has_color_card': False,
        'has_blank_ref': True,
        'known_concs': '0,1,2,5,10,20,50',
        'replicates': '3',
        'image_count': '21',
        'note': '单卡片槽',
    }
    r = client.put('/api/experiment', json=payload)
    assert r.status_code == 200
    saved = client.get('/api/experiment').get_json()
    assert saved['analyte'] == '某农药'
    assert saved['known_concs'] == '0,1,2,5,10,20,50'
    assert saved['replicates'] == '3'


def test_p1_validate_roi_names(client):
    assert client.put('/api/experiment', json={'roi_names': []}).status_code == 400
    assert client.put('/api/experiment', json={'roi_names': ['T', 'T']}).status_code == 400
    assert client.put('/api/experiment', json={'roi_names': ['']}).status_code == 400


def test_p1_validate_roi_count(client):
    assert client.put('/api/experiment', json={'roi_count': 0}).status_code == 400
    assert client.put('/api/experiment', json={'roi_count': 11}).status_code == 400
    assert client.put('/api/experiment', json={'roi_count': 'x'}).status_code == 400


def test_p1_validate_concs(client):
    assert client.put('/api/experiment', json={'known_concs': '0,a,3'}).status_code == 400
    assert client.put('/api/experiment', json={'known_concs': '0,-1,3'}).status_code == 400
    r = client.put('/api/experiment', json={'known_concs': '0, 1.5, 2'})
    assert r.status_code == 200
    assert r.get_json()['experiment']['known_concs'] == '0,1.5,2'


def test_p1_validate_replicates(client):
    assert client.put('/api/experiment', json={'replicates': 'x'}).status_code == 400
    assert client.put('/api/experiment', json={'replicates': '0'}).status_code == 400
    assert client.put('/api/experiment', json={'replicates': '3'}).status_code == 200


def test_p1_persists_in_settings(client):
    client.put('/api/experiment', json={'analyte': '持久化测试', 'roi_names': ['T', 'Bg']})
    # 直接查 settings 表：重启后仍可读（数据持久化，不丢）
    dpath = Path(client.application.config['DATA_DIR']) / 'fluro.db'
    row = db.query_one(dpath, "SELECT value FROM settings WHERE key='experiment'")
    assert row is not None
    assert '持久化测试' in row['value']


def test_p1_roi_names_saved_for_modeling(client):
    """ROI 命名（T/Bg）是建模自动识别的基础，保存后必须完整读回。"""
    client.put('/api/experiment', json={'roi_names': ['T', 'Bg']})
    ex = client.get('/api/experiment').get_json()
    assert ex['roi_names'] == ['T', 'Bg']


# ============ 问题 2-A：多命名 ROI 引擎（T/Bg） ============

ROIS_T_BG = [
    {'name': 'T', 'role': 'sample', 'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44, 'bg_subtract': 0},
    {'name': 'Bg', 'role': 'background', 'x': 0.10, 'y': 0.10, 'w': 0.20, 'h': 0.30, 'bg_subtract': 0},
]


def _upload_img(client, conc=50, kind='detection'):
    import io
    import cv2 as _cv2
    from tools import make_test_images as _mti
    img = _mti.make_test_image(conc)
    ok, buf = _cv2.imencode('.png', img[:, :, ::-1])
    data = {'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': kind}
    if kind == 'calibration':
        data['known_conc'] = str(conc)
    return client.post('/api/images/upload', data=data, content_type='multipart/form-data').get_json()['images'][0]['id']


def test_p2a_save_and_get_multi_rois(client):
    iid = _upload_img(client)
    r = client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
    assert r.status_code == 200
    got = client.get(f'/api/images/{iid}/rois').get_json()['rois']
    assert len(got) == 2
    names = sorted(x['name'] for x in got)
    assert names == ['Bg', 'T']
    t = next(x for x in got if x['name'] == 'T')
    assert t['role'] == 'sample'
    bg = next(x for x in got if x['name'] == 'Bg')
    assert bg['role'] == 'background'


def test_p2a_multi_roi_validation(client):
    iid = _upload_img(client)
    assert client.post(f'/api/images/{iid}/rois', json={'rois': []}).status_code == 400
    dup = [dict(ROIS_T_BG[0]), dict(ROIS_T_BG[0])]
    dup[0]['name'] = 'T'
    assert client.post(f'/api/images/{iid}/rois', json={'rois': dup}).status_code == 400
    bad = [dict(ROIS_T_BG[0], x=1.5)]
    assert client.post(f'/api/images/{iid}/rois', json={'rois': bad}).status_code == 400
    missing = [{'name': 'T'}]
    assert client.post(f'/api/images/{iid}/rois', json={'rois': missing}).status_code == 400


def test_p2a_main_roi_synced_to_legacy(client):
    """多 ROI 保存后，主检测区（T）同步到旧 roi 表，兼容原有特征/检测链路。"""
    iid = _upload_img(client)
    client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
    legacy = client.get(f'/api/pipeline/{iid}/roi').get_json()['roi']
    assert legacy is not None
    assert abs(legacy['x'] - ROIS_T_BG[0]['x']) < 1e-6


def test_p2a_features_use_main_roi(client):
    iid = _upload_img(client, conc=50)
    client.post('/api/pipeline/run', json={'image_ids': [iid]})
    client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
    f = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    assert f['mean_r'] is not None and f['hue'] is not None


def test_p2a_template_multi_roi_apply(client):
    iid = _upload_img(client)
    tpl = {
        'name': '卡片模板',
        'template_json': {
            'T': {'x': 0.335, 'y': 0.28, 'w': 0.33, 'h': 0.44, 'role': 'sample'},
            'Bg': {'x': 0.10, 'y': 0.10, 'w': 0.20, 'h': 0.30, 'role': 'background'},
        },
    }
    tid = client.post('/api/templates', json=tpl).get_json()['id']
    r = client.post(f'/api/images/{iid}/rois/apply_template', json={'template_id': tid})
    assert r.status_code == 200
    got = client.get(f'/api/images/{iid}/rois').get_json()['rois']
    assert len(got) == 2 and sorted(x['name'] for x in got) == ['Bg', 'T']
    assert all(x['source'] == 'template' for x in got)


def test_p2a_apply_legacy_single_template_as_t(client):
    iid = _upload_img(client)
    tid = client.post('/api/templates', json={'name': '单ROI', 'x': 0.3, 'y': 0.3, 'w': 0.4, 'h': 0.4}).get_json()['id']
    r = client.post(f'/api/images/{iid}/rois/apply_template', json={'template_id': tid})
    assert r.status_code == 200
    got = client.get(f'/api/images/{iid}/rois').get_json()['rois']
    assert len(got) == 1 and got[0]['name'] == 'T'


def test_p2a_overlay_all_rois(client):
    iid = _upload_img(client)
    client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
    r = client.get(f'/api/images/{iid}/overlay')
    assert r.status_code == 200 and r.headers['Content-Type'].startswith('image/png')


# ============ 问题 2-B：扩展特征提取（RGB/SD、Lab、OD、ΔE、比值） ============

def test_p2b_extract_roi_features_full_fields(client):
    from app.image_processing import extract_roi_features
    import numpy as _np
    from tools import make_test_images as _mti
    img = _mti.make_test_image(50)
    f = extract_roi_features(img, (0.335, 0.28, 0.33, 0.44))
    for key in ('mean_r', 'mean_g', 'mean_b', 'median_r', 'median_g', 'median_b',
                'std_r', 'std_g', 'std_b', 'hue', 'saturation', 'value',
                'lab_l', 'lab_a', 'lab_b', 'gray', 'od_r', 'od_g', 'od_b',
                'ratio_gr', 'ratio_gb', 'ratio_rb'):
        assert f[key] is not None, f'{key} 缺失'
    # OD 单调性：亮像素 OD 小，暗像素 OD 大
    bright = _np.full((20, 20, 3), 240, dtype=_np.uint8)
    dark = _np.full((20, 20, 3), 30, dtype=_np.uint8)
    assert extract_roi_features(bright, (0, 0, 1, 1))['od_r'] < extract_roi_features(dark, (0, 0, 1, 1))['od_r']


def test_p2b_compute_multi_roi_features(client):
    iid = _upload_img(client, conc=100)  # T 区红色、Bg 区取背景附近
    client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
    r = client.post(f'/api/images/{iid}/features', json={})
    assert r.status_code == 200
    d = r.get_json()
    assert 'T' in d['rois'] and 'Bg' in d['rois']
    t = d['rois']['T']
    assert t['lab_l'] is not None and t['od_g'] is not None
    comb = d['combined']
    assert comb.get('T_R_over_Bg_R') is not None
    assert comb.get('T_R_minus_Bg_R') is not None
    assert comb.get('deltaE_T_vs_Bg') is not None
    # 主 ROI 兼容旧特征表
    legacy = client.get(f'/api/pipeline/{iid}/features').get_json()['features']
    assert legacy['hue'] is not None


def test_p2b_roi_features_persisted(client):
    iid = _upload_img(client, conc=50)
    client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
    client.post(f'/api/images/{iid}/features', json={})
    got = client.get(f'/api/images/{iid}/features').get_json()
    assert set(got['rois'].keys()) == {'T', 'Bg'}
    assert 'deltaE_T_vs_Bg' in got['combined']
    assert got['combined']['T_G_over_Bg_G'] > 0


def test_p2b_derive_delta_e_symmetric(client):
    """色差 ΔE 应为对称且非负（Lab 空间欧氏距离）。"""
    from app.image_processing import derive_combined_features
    f1 = {'mean_r': 200, 'mean_g': 50, 'mean_b': 50, 'od_r': 0.1, 'od_g': 0.7, 'od_b': 0.7,
          'lab_l': 60, 'lab_a': 70, 'lab_b': 40}
    f2 = {'mean_r': 90, 'mean_g': 90, 'mean_b': 90, 'od_r': 0.5, 'od_g': 0.5, 'od_b': 0.5,
          'lab_l': 70, 'lab_a': 0, 'lab_b': 0}
    c1 = derive_combined_features({'T': f1, 'Bg': f2})
    c2 = derive_combined_features({'T': f2, 'Bg': f1})
    assert c1['deltaE_T_vs_Bg'] == c2['deltaE_T_vs_Bg']
    assert c1['deltaE_T_vs_Bg'] > 0


def test_p2b_features_require_roi(client):
    iid = _upload_img(client)
    r = client.post(f'/api/images/{iid}/features', json={})
    assert r.status_code == 400
    assert 'ROI' in r.get_json()['error']
