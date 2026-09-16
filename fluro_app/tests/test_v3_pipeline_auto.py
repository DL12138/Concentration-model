# -*- coding: utf-8 -*-
"""问题2：上传页「自动处理全部」后，直接完成其余工作流并产出可显示的结果数据。

验证 pipeline/run 的接口契约（前端 runPipeline 依赖）：
- 已有生效模型时：检测图 result='ok'，检测记录已写入，结果数据可取；
- 无生效模型时：result='attention' 且不产出检测记录（前端据此提示去建模）。
"""
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


def _upload(client, kind='detection'):
    from tools import make_test_images as mti
    import cv2
    img = mti.make_test_image(50)
    ok, buf = cv2.imencode('.png', img[:, :, ::-1])
    return client.post('/api/images/upload',
                       data={'files': [(io.BytesIO(buf.tobytes()), 't.png')], 'kind': kind},
                       content_type='multipart/form-data').get_json()['images'][0]['id']


def _make_calib_csv(rows):
    s = 'concentration,T_R_over_Bg_R\n' + ''.join(f'{a},{b}\n' for a, b in rows)
    return io.BytesIO(s.encode('utf-8'))


def _save_model(client):
    rows = [(0, 1.0), (1, 0.91), (2, 0.82), (5, 0.55), (10, 0.1)]
    client.post('/api/modeling/import',
                data={'file': (_make_calib_csv(rows), 'cal.csv'),
                      'conc_col': 'concentration', 'feature_col': 'T_R_over_Bg_R'},
                content_type='multipart/form-data')
    fit = client.post('/api/calibration/fit', json={'feature': 'T_R_over_Bg_R'}).get_json()
    lin = fit['results']['linear']
    client.post('/api/models', json={
        'name': '自动流程模型', 'type': 'linear', 'params': lin['params'],
        'metrics': {'r2': lin['r2']},
        'source_snapshot': {'feature': 'T_R_over_Bg_R', 'data': fit['data'], 'n': fit['n'],
                            'preprocess': fit['preprocess']},
    })


def test_pipeline_auto_full_to_result(client):
    """有生效模型 + ROI 模板：点击自动处理后直接产出检测结果（result=ok，记录可查）。"""
    _save_model(client)
    # 保存 ROI 模板（单卡片：T 检测区 + Bg 背景区），流水线自动套用
    client.post('/api/templates', json={'name': '单卡片模板', 'template_json': {
        'T': {'x': 0.325, 'y': 0.267, 'w': 0.35, 'h': 0.467, 'role': 'sample'},
        'Bg': {'x': 0.1, 'y': 0.133, 'w': 0.25, 'h': 0.5, 'role': 'background'},
    }})
    iid = _upload(client)
    # 自动处理全部（等价上传页「自动处理全部」按钮）
    r = client.post('/api/pipeline/run', json={'image_ids': [iid]})
    assert r.status_code == 200
    steps = r.get_json()['results'][0]['steps']
    # 每步都完成（ROI 无模板时为 attention 需人工框选；结果必须 ok）
    assert steps['preprocess'] == 'ok'
    assert steps['channels'] == 'ok'
    assert steps['result'] == 'ok'
    # 结果数据已写入：检测记录存在且可取
    dets = client.get('/api/detections').get_json()['detections']
    assert any(d['image_id'] == iid and d['conc'] is not None for d in dets)
    det = client.post(f'/api/detect/{iid}', json={}).get_json()
    assert det['ok'] is True
    assert det['detection']['conc'] is not None


def test_pipeline_auto_attention_without_model(client):
    """无生效模型：自动处理后 result=attention，无检测记录（前端提示先建模）。"""
    iid = _upload(client)
    r = client.post('/api/pipeline/run', json={'image_ids': [iid]})
    steps = r.get_json()['results'][0]['steps']
    assert steps['result'] == 'attention'
    dets = client.get('/api/detections').get_json()['detections']
    assert all(d['image_id'] != iid for d in dets)


def test_pipeline_auto_with_manual_roi_fix(client):
    """自动处理无模板时若未识别出检测区，人工框选后可重跑并产出结果。"""
    _save_model(client)
    iid = _upload(client)
    r1 = client.post('/api/pipeline/run', json={'image_ids': [iid]})
    s1 = r1.get_json()['results'][0]['steps']
    if s1['roi'] == 'attention':
        # 自动识别失败 → 人工框选 ROI（归一化坐标）
        client.post(f'/api/images/{iid}/rois', json={'rois': ROIS_T_BG})
    # 重跑整条流水线，应全绿并产出结果
    r2 = client.post('/api/pipeline/run', json={'image_ids': [iid]})
    steps = r2.get_json()['results'][0]['steps']
    assert steps['roi'] == 'ok'
    assert steps['result'] == 'ok'
