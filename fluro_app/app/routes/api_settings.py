# -*- coding: utf-8 -*-
"""设置与备份 API（M8）：检测上下限、单位、批次、参考图、模板/模型管理、备份恢复。"""
import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, abort

from .. import database as db
from .. import backup as backup_mod

bp = Blueprint('settings', __name__, url_prefix='/api')


def _db_path():
    return Path(current_app.config['DATA_DIR']) / 'fluro.db'


# 单卡片槽比色检测：实验信息默认配置（问题1）
DEFAULT_EXPERIMENT = {
    'analyte': '',               # 待测物（待用户填写）
    'color_trend': '',           # 颜色变化趋势（浓度越高 T 区越…）
    'carrier': 'single_card',    # 检测载体：单卡片/试纸条
    'roi_count': 2,              # 每张图 ROI 数量
    'roi_names': ['T', 'Bg'],    # ROI 命名（样品检测区 T、背景区 Bg）
    'has_color_card': False,     # 是否有色卡
    'has_blank_ref': True,       # 是否有空白参考区（背景区 Bg）
    'known_concs': '',           # 已知浓度点（逗号分隔，如 0,1,2,5,10,20,50）
    'replicates': '',            # 每个浓度重复次数
    'image_count': '',           # 图片数量
    'note': '',
}


@bp.route('/experiment', methods=['GET'])
def get_experiment():
    raw = db.get_setting(_db_path(), 'experiment')
    if raw:
        try:
            val = json.loads(raw)
            if isinstance(val, dict) and 'roi_names' in val:
                return jsonify(val)
        except Exception:  # noqa: BLE001
            pass
    return jsonify(DEFAULT_EXPERIMENT)


@bp.route('/experiment', methods=['PUT'])
def save_experiment():
    body = request.get_json(silent=True) or {}
    exp = dict(DEFAULT_EXPERIMENT)
    for k in exp:
        if k in body:
            exp[k] = body[k]
    # ROI 名称校验：至少 1 个、不重复、1-12 字符
    names = exp.get('roi_names')
    if not isinstance(names, list) or not names:
        return jsonify({'error': 'ROI 名称至少一个'}), 400
    names = [str(n).strip() for n in names if str(n).strip()]
    if not names:
        return jsonify({'error': 'ROI 名称至少一个'}), 400
    if len(names) != len(set(names)):
        return jsonify({'error': 'ROI 名称不能重复'}), 400
    for n in names:
        if not (1 <= len(n) <= 12):
            return jsonify({'error': 'ROI 名称须为 1-12 个字符'}), 400
    exp['roi_names'] = names
    # ROI 数量校验
    try:
        rc = int(exp.get('roi_count'))
    except (TypeError, ValueError):
        return jsonify({'error': 'ROI 数量必须为整数'}), 400
    if not (1 <= rc <= 10):
        return jsonify({'error': 'ROI 数量须在 1-10 之间'}), 400
    exp['roi_count'] = rc
    # 浓度点校验：逗号分隔非负数字，允许为空
    if exp.get('known_concs'):
        parts = [p.strip() for p in str(exp['known_concs']).split(',') if p.strip()]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return jsonify({'error': '浓度点格式错误：用逗号分隔数字，如 0,1,2,5'}), 400
        if any(v < 0 for v in vals):
            return jsonify({'error': '浓度点不能为负数'}), 400
        exp['known_concs'] = ','.join(parts)
    # 重复次数校验
    if exp.get('replicates'):
        try:
            rep = int(exp['replicates'])
        except (TypeError, ValueError):
            return jsonify({'error': '重复次数必须为整数'}), 400
        if rep < 1:
            return jsonify({'error': '重复次数至少为 1'}), 400
        exp['replicates'] = str(rep)
    for k in ('analyte', 'color_trend', 'carrier', 'image_count', 'note'):
        exp[k] = str(exp.get(k) or '').strip()
    db.set_setting(_db_path(), 'experiment', json.dumps(exp, ensure_ascii=False))
    return jsonify({'ok': True, 'experiment': exp})


@bp.route('/settings', methods=['GET'])
def get_settings():
    keys = ['limit_lower', 'limit_upper', 'unit', 'current_batch', 'default_filter',
            'default_kernel', 'dark_ref_path', 'flat_ref_path']
    out = {}
    for k in keys:
        out[k] = db.get_setting(_db_path(), k)
    refs = {}
    try:
        from .api_workflow import get_refs as _refs
        refs = _refs().get_json()
    except Exception:  # noqa: BLE001
        pass
    templates = db.query(_db_path(), 'SELECT * FROM roi_templates ORDER BY id DESC')
    models = db.query(_db_path(), 'SELECT * FROM models ORDER BY is_active DESC, id DESC')
    backups = backup_mod.list_backups(current_app.config['DATA_DIR'])
    return jsonify({
        'settings': out,
        'refs': refs,
        'templates': templates,
        'models': models,
        'backups': backups,
        'data_dir': str(current_app.config['DATA_DIR']),
    })


@bp.route('/settings', methods=['POST'])
def save_settings():
    """批量保存设置。body: {settings: {key: value}}；value=None 表示清除。"""
    body = request.get_json(silent=True) or {}
    st = body.get('settings') or {}
    if not isinstance(st, dict):
        return jsonify({'error': 'settings 必须为对象'}), 400
    for k, v in st.items():
        if v is None:
            db.execute(_db_path(), 'DELETE FROM settings WHERE key=?', (k,))
        else:
            db.set_setting(_db_path(), k, str(v))
    return jsonify({'ok': True})


@bp.route('/backup', methods=['POST'])
def create_backup():
    body = request.get_json(silent=True) or {}
    try:
        b = backup_mod.create_backup(current_app.config['DATA_DIR'], label=body.get('label'))
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'backup': b})


@bp.route('/backups', methods=['GET'])
def list_backups():
    return jsonify({'backups': backup_mod.list_backups(current_app.config['DATA_DIR'])})


@bp.route('/backup/restore', methods=['POST'])
def restore_backup():
    body = request.get_json(silent=True) or {}
    name = body.get('name')
    if not name:
        return jsonify({'error': '缺少备份名称'}), 400
    try:
        r = backup_mod.restore_backup(current_app.config['DATA_DIR'], name)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    # 恢复后重新初始化数据库结构（迁移幂等）
    db.init_db(_db_path())
    return jsonify({'ok': True, **r})


@bp.route('/backup/<path:name>', methods=['DELETE', 'POST'])
def delete_backup(name):
    from .. import backup as bm
    backups_dir = bm._default_backups_dir(current_app.config['DATA_DIR'])
    target = backups_dir / name
    if not target.exists() or not target.is_dir():
        abort(404)
    import shutil
    shutil.rmtree(target)
    return jsonify({'ok': True})


@bp.route('/backup/<path:name>/delete', methods=['POST'])
def delete_backup_alias(name):
    return delete_backup(name)
