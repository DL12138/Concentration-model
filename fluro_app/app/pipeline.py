# -*- coding: utf-8 -*-
"""流水线编排（M4）：上传后自动依次执行 预处理 → ROI → 特征 → 结果。

每张图独立执行，任一上游步骤失败/待人工时标记为 attention（不阻断整批）。
"""
from flask import current_app

from . import database as db
from .config import Config


def _db_path():
    from pathlib import Path
    return Path(current_app.config['DATA_DIR']) / 'fluro.db'


def run_single(image_id):
    """执行单张图的完整流水线。返回该图各步骤状态。"""
    from .routes.api_workflow import preprocess_core, auto_roi_core, compute_features_core, channels_core
    from .routes.api_model import run_detection

    result = {'image_id': image_id, 'steps': {}, 'status': 'ok'}

    # 1. 预处理（默认参数）
    try:
        preprocess_core(image_id, Config.DEFAULT_FILTER, Config.DEFAULT_KERNEL, False, False)
        result['steps']['preprocess'] = 'ok'
    except KeyError:
        result['steps']['preprocess'] = 'error'
        result['status'] = 'error'
        return result
    except Exception as e:  # noqa: BLE001
        result['steps']['preprocess'] = 'error'
        result['status'] = 'attention'

    # 1.5 通道分离（问题5 新增步骤：预处理后、ROI 前）
    try:
        channels_core(image_id)
        result['steps']['channels'] = 'ok'
    except Exception:  # noqa: BLE001
        result['steps']['channels'] = 'attention'

    # 2. ROI（有激活模板则自动套用）
    roi_ok = False
    try:
        auto_roi_core(image_id)
        result['steps']['roi'] = 'ok'
        roi_ok = True
    except ValueError:
        result['steps']['roi'] = 'attention'   # 无模板，需人工框选
    except Exception:  # noqa: BLE001
        result['steps']['roi'] = 'attention'

    # 3. 特征（ROI 就绪才计算）
    if roi_ok:
        try:
            compute_features_core(image_id)
            result['steps']['feature'] = 'ok'
        except Exception:  # noqa: BLE001
            result['steps']['feature'] = 'attention'
    else:
        result['steps']['feature'] = 'attention'

    # 4. 结果：检测图 → 真实检测；标定图 → 待加入标定数据集（pending，不算异常）
    row = db.query_one(_db_path(), 'SELECT kind FROM images WHERE id=?', (image_id,))
    if row and row['kind'] == 'calibration':
        result['steps']['result'] = 'pending'
    else:
        try:
            run_detection(image_id)
            result['steps']['result'] = 'ok'
        except ValueError:
            result['steps']['result'] = 'attention'   # 无生效模型等
        except Exception:  # noqa: BLE001
            result['steps']['result'] = 'attention'

    if any(v == 'attention' for v in result['steps'].values()):
        result['status'] = 'attention'
    elif any(v == 'error' for v in result['steps'].values()):
        result['status'] = 'error'
    else:
        result['status'] = 'ok'

    db.execute(_db_path(), 'UPDATE images SET status=? WHERE id=?', (result['status'], image_id))
    return result


def run_batch(image_ids=None):
    """批量执行流水线。image_ids 为空时处理所有 uploaded 状态的图。"""
    if not image_ids:
        rows = db.query(_db_path(), "SELECT id FROM images WHERE status='uploaded' ORDER BY id")
        image_ids = [r['id'] for r in rows]
    results = [run_single(iid) for iid in image_ids]
    return results
