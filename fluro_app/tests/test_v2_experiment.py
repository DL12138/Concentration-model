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
