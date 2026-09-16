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
