# -*- coding: utf-8 -*-
"""标定建模 API（M5）：浓度分组、数据点纳入/剔除、多模型拟合、模型库管理。"""
import csv
import io
import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, abort, send_file

from .. import database as db
from .. import modeling
from ..config import Config

bp = Blueprint('model', __name__, url_prefix='/api')


def _db_path():
    return Path(current_app.config['DATA_DIR']) / 'fluro.db'


def recompute_downstream(image_id):
    """ROI/预处理变更后的级联重算：重算特征；已检测过的图重算检测结果。"""
    try:
        from ..routes.api_workflow import compute_features_core
        compute_features_core(image_id)
    except (KeyError, ValueError):
        pass  # 无 ROI/图时跳过，等待后续步骤补齐
    try:
        existing = db.query_one(_db_path(), 'SELECT id FROM detections WHERE image_id=?', (image_id,))
        if existing:
            run_detection(image_id)
    except (KeyError, ValueError):
        pass


def _collect_calibration_data(feature=None):
    """收集全部标定数据点：返回 (xs, ys, points)，points 含各分组信息。

    支持两类数据点：
    - 应用内（source='app'）：image_id 指向真实图，特征取自 features 表；
    - 导入（source='import'）：image_id=0，特征取自 feature_value（问题2-G CSV/Excel 导入）。
    """
    rows = db.query(_db_path(),
                    'SELECT cp.id AS point_id, cp.group_id, cp.included, cp.image_id, '
                    'cp.feature_value, cp.feature_name, cp.source, '
                    'cg.conc, cg.name AS group_name, cg.unit, i.replicate AS rep, f.* '
                    'FROM calibration_points cp '
                    'JOIN calibration_groups cg ON cg.id = cp.group_id '
                    'LEFT JOIN images i ON i.id = cp.image_id '
                    'LEFT JOIN features f ON f.image_id = cp.image_id '
                    'ORDER BY cg.conc, cp.id')
    groups_map = {}
    xs, ys, points = [], [], []
    for r in rows:
        groups_map.setdefault(r['group_id'], {
            'id': r['group_id'], 'name': r['group_name'], 'conc': r['conc'],
            'unit': r.get('unit') or 'ng/mL', 'points': [],
        })
        feat = r.get('feature_value')
        if feat is None:
            feat = r.get(feature) if feature else r.get('hue')
        feats = {k: r[k] for k in modeling.FEATURES if k in r}
        # 导入/自定义特征（如 T_R_over_Bg_R）也纳入 features，便于组统计
        if r.get('feature_name') and r.get('feature_value') is not None:
            feats.setdefault(str(r['feature_name']).strip(), r['feature_value'])
        groups_map[r['group_id']]['points'].append({
            'point_id': r['point_id'], 'image_id': r['image_id'], 'included': r['included'],
            'source': r.get('source') or 'app',
            'replicate': r.get('rep'),
            'features': feats,
            'feature_value': r.get('feature_value'),
        })
        if feature:
            points.append(r)
            if r['included']:
                xs.append(r['conc'])
                ys.append(feat)
    groups = list(groups_map.values())
    return xs, ys, groups


# ---- 问题2-G：CSV/Excel 标定数据导入 ----

def _parse_table_file(file_storage):
    """解析 CSV / Excel（.csv/.xlsx/.xls），返回 (columns, rows)。"""
    filename = file_storage.filename or ''
    suffix = Path(filename).suffix.lower()
    if suffix == '.csv':
        import csv as _csv
        import io as _io
        raw = file_storage.read().decode('utf-8-sig', errors='replace')
        reader = _csv.DictReader(_io.StringIO(raw))
        columns = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    elif suffix in ('.xlsx', '.xls'):
        try:
            import openpyxl
        except ImportError:
            raise ValueError('未安装 openpyxl，请运行 pip install openpyxl')
        wb = openpyxl.load_workbook(_io_bytes(file_storage), data_only=True)
        ws = wb.active
        all_rows = list(ws.iter_rows(values_only=True))
        if not all_rows:
            return [], []
        header = [str(c).strip() if c is not None else '' for c in all_rows[0]]
        columns = header
        rows = []
        for line in all_rows[1:]:
            row = {}
            for i, col in enumerate(header):
                if col:
                    row[col] = line[i] if i < len(line) else None
            rows.append(row)
    else:
        raise ValueError('仅支持 .csv / .xlsx / .xls 文件')
    return columns, rows


def _io_bytes(file_storage):
    import io as _io
    return _io.BytesIO(file_storage.read())


@bp.route('/modeling/import_preview', methods=['POST'])
def modeling_import_preview():
    """解析上传的 CSV/Excel，返回列名与前几行供选择浓度列/特征列。"""
    f = request.files.get('file')
    if not f:
        return jsonify({'error': '未收到文件'}), 400
    try:
        columns, rows = _parse_table_file(f)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if not columns:
        return jsonify({'error': '文件为空或缺少表头'}), 400
    return jsonify({'ok': True, 'columns': columns, 'preview': rows[:5], 'total_rows': len(rows)})


@bp.route('/modeling/import', methods=['POST'])
def modeling_import():
    """导入 CSV/Excel 标定数据：按浓度分组建立标定组（重复行=同浓度重复点）。"""
    f = request.files.get('file')
    conc_col = (request.form.get('conc_col') or '').strip()
    feat_col = (request.form.get('feature_col') or '').strip()
    unit = (request.form.get('unit') or '').strip() or 'ng/mL'
    if unit not in Config.CONC_UNITS:
        return jsonify({'error': f'不支持的浓度单位：{unit}'}), 400
    if not f:
        return jsonify({'error': '未收到文件'}), 400
    if not conc_col or not feat_col:
        return jsonify({'error': '请选择浓度列与特征列'}), 400
    try:
        columns, rows = _parse_table_file(f)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if conc_col not in columns or feat_col not in columns:
        return jsonify({'error': f'列不存在：{conc_col} / {feat_col}'}), 400

    groups = {}
    imported = 0
    skipped = 0
    for row in rows:
        try:
            conc = float(row.get(conc_col))
        except (TypeError, ValueError):
            skipped += 1
            continue
        feat_raw = row.get(feat_col)
        try:
            feat = float(feat_raw)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if conc not in groups:
            gid = db.execute(_db_path(),
                             'INSERT INTO calibration_groups (name, conc, unit) VALUES (?,?,?)',
                             (f'{Path(f.filename).stem} @ {conc}', conc, unit))
            groups[conc] = gid
        db.execute(_db_path(),
                   'INSERT INTO calibration_points (group_id, image_id, included, feature_value, feature_name, source) '
                   'VALUES (?,?,1,?,?,?)',
                   (groups[conc], 0, feat, feat_col, 'import'))
        imported += 1
    if imported == 0:
        return jsonify({'error': f'没有可导入的数据行（浓度/特征列 {conc_col}/{feat_col} 无法解析）'}), 400
    return jsonify({'ok': True, 'imported': imported, 'skipped': skipped, 'groups': len(groups)})


@bp.route('/calibration/data', methods=['GET'])
def calibration_data():
    xs, ys, groups = _collect_calibration_data()
    # 每组均值与标准差（included 点）
    for g in groups:
        inc = [p['features'] for p in g['points'] if p['included']]
        if not inc:
            g['mean'] = None
            g['sd'] = None
            g['n'] = 0
            continue
        means = {}
        sds = {}
        keys = list(modeling.FEATURES)
        for p in inc:
            for fn in p:
                if fn not in keys:
                    keys.append(fn)
        for k in keys:
            vals = [p[k] for p in inc if p.get(k) is not None]
            if vals:
                means[k] = round(sum(vals) / len(vals), 4)
                sds[k] = round((sum((v - means[k]) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5, 4) if len(vals) > 1 else None
        cvs = {}
        for k in means:
            cvs[k] = round(sds[k] / means[k] * 100, 2) if means[k] and sds[k] is not None else None
        g['mean'] = means
        g['sd'] = sds
        g['cv'] = cvs
        g['n'] = len(inc)
    active = db.query_one(_db_path(), 'SELECT * FROM models WHERE is_active=1 ORDER BY id DESC LIMIT 1')
    models = db.query(_db_path(), 'SELECT * FROM models ORDER BY id DESC')
    # 待分组的标定图：kind=calibration、已有特征、尚未加入任何分组
    pending = db.query(_db_path(),
                       'SELECT i.id, i.known_conc, f.hue FROM images i '
                       'JOIN features f ON f.image_id=i.id '
                       'LEFT JOIN calibration_points cp ON cp.image_id=i.id '
                       "WHERE i.kind='calibration' AND cp.id IS NULL "
                       'ORDER BY i.id DESC')
    return jsonify({'groups': groups, 'active_model': active, 'models': models, 'pending_cal': pending})


@bp.route('/calibration/groups', methods=['POST'])
def create_group():
    body = request.get_json(silent=True) or {}
    try:
        conc = float(body['conc'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': '浓度必须为数字'}), 400
    name = (body.get('name') or '').strip() or f'C{conc:g}'
    # 同浓度复用
    existing = db.query_one(_db_path(), 'SELECT id FROM calibration_groups WHERE conc=?', (conc,))
    if existing:
        return jsonify({'ok': True, 'id': existing['id'], 'reused': True})
    gid = db.execute(_db_path(),
                     'INSERT INTO calibration_groups (name, conc) VALUES (?,?)', (name, conc))
    return jsonify({'ok': True, 'id': gid, 'reused': False})


@bp.route('/calibration/groups/<int:gid>', methods=['DELETE', 'POST'])
def delete_group(gid):
    db.execute(_db_path(), 'DELETE FROM calibration_points WHERE group_id=?', (gid,))
    db.execute(_db_path(), 'DELETE FROM calibration_groups WHERE id=?', (gid,))
    return jsonify({'ok': True})


@bp.route('/calibration/groups/<int:gid>/delete', methods=['POST'])
def delete_group_alias(gid):
    return delete_group(gid)


@bp.route('/calibration/points', methods=['POST'])
def add_point():
    """把一张标定图加入浓度分组。"""
    body = request.get_json(silent=True) or {}
    image_id = body.get('image_id')
    group_id = body.get('group_id')
    if not image_id or not group_id:
        return jsonify({'error': '需要 image_id 与 group_id'}), 400
    img = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    grp = db.query_one(_db_path(), 'SELECT * FROM calibration_groups WHERE id=?', (group_id,))
    if not img or not grp:
        return jsonify({'error': '图片或分组不存在'}), 404
    feat = db.query_one(_db_path(), 'SELECT * FROM features WHERE image_id=?', (image_id,))
    if not feat:
        return jsonify({'error': '该图尚无特征，请先完成 ROI 与特征提取'}), 400
    dup = db.query_one(_db_path(), 'SELECT id FROM calibration_points WHERE image_id=?', (image_id,))
    if dup:
        return jsonify({'ok': True, 'id': dup['id'], 'reused': True})
    pid = db.execute(_db_path(),
                     'INSERT INTO calibration_points (group_id, image_id, included) VALUES (?,?,1)',
                     (group_id, image_id))
    return jsonify({'ok': True, 'id': pid, 'reused': False})


@bp.route('/calibration/quick', methods=['POST'])
def quick_calibrate():
    """一键标定：把一张已保存结果的标定图，按给定浓度加入（或复用）浓度分组。

    供「结果界面 → 写浓度 → 加入标定数据集」链路使用。返回 group_id/point_id。
    """
    body = request.get_json(silent=True) or {}
    image_id = body.get('image_id')
    try:
        conc = float(body.get('conc'))
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': '浓度必须为数字'}), 400
    img = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not img:
        return jsonify({'error': '图片不存在'}), 404
    if img['kind'] != 'calibration':
        return jsonify({'error': '仅标定图可加入标定数据集'}), 400
    feat = db.query_one(_db_path(), 'SELECT * FROM features WHERE image_id=?', (image_id,))
    if not feat:
        return jsonify({'error': '该图尚无特征，请先完成 ROI 与特征提取'}), 400
    unit = (img.get('conc_unit') or '').strip() or 'ng/mL'
    grp = db.query_one(_db_path(), 'SELECT id FROM calibration_groups WHERE conc=?', (conc,))
    if grp:
        gid = grp['id']
        db.execute(_db_path(), 'UPDATE calibration_groups SET unit=? WHERE id=?', (unit, gid))
    else:
        gid = db.execute(_db_path(),
                         'INSERT INTO calibration_groups (name, conc, unit) VALUES (?,?,?)',
                         (f'C{conc:g}', conc, unit))
    dup = db.query_one(_db_path(), 'SELECT id FROM calibration_points WHERE image_id=?', (image_id,))
    if dup:
        db.execute(_db_path(),
                   'UPDATE calibration_points SET group_id=?, included=1 WHERE id=?',
                   (gid, dup['id']))
        pid, reused = dup['id'], True
    else:
        pid = db.execute(_db_path(),
                         'INSERT INTO calibration_points (group_id, image_id, included) VALUES (?,?,1)',
                         (gid, image_id))
        reused = False
    return jsonify({'ok': True, 'group_id': gid, 'point_id': pid, 'conc': conc, 'unit': unit, 'reused': reused})


@bp.route('/calibration/points/<int:pid>', methods=['POST'])
def toggle_point(pid):
    """剔除/纳入数据点。body: {included: 0|1}"""
    row = db.query_one(_db_path(), 'SELECT * FROM calibration_points WHERE id=?', (pid,))
    if not row:
        abort(404)
    body = request.get_json(silent=True) or {}
    inc = 1 if body.get('included', 1) else 0
    db.execute(_db_path(), 'UPDATE calibration_points SET included=? WHERE id=?', (inc, pid))
    return jsonify({'ok': True, 'id': pid, 'included': inc})


@bp.route('/calibration/fit', methods=['POST'])
def fit_calibration():
    """用当前 included 数据点拟合 4 类模型。body: {feature: 'hue'}

    数据先应用建模预处理（问题2-H：IQR 剔除 / log 浓度 / Z-score 标准化）。
    """
    body = request.get_json(silent=True) or {}
    feature = body.get('feature', 'hue')
    if feature not in modeling.FEATURES:
        # 问题2-G：允许用 CSV/Excel 导入的自定义特征列（如 T_R_over_Bg_R）
        imported = db.query_one(_db_path(),
                                "SELECT 1 FROM calibration_points WHERE source='import' AND feature_name=? LIMIT 1",
                                (feature,))
        if not imported:
            return jsonify({'error': f'不支持的特征：{feature}（可选：{", ".join(modeling.FEATURES)}）'}), 400
    xs, ys, groups = _collect_calibration_data(feature)
    if len(xs) < 3 or len(set(xs)) < 3:
        return jsonify({'error': '有效标定点不足（至少 3 个不同浓度、每浓度有特征数据）'}), 400
    prep = _preprocess_settings()
    xs, ys, prep_meta = modeling.preprocess_data(xs, ys, log_conc=prep['log_conc'],
                                                 zscore=prep['zscore'], iqr=prep['iqr'])
    prep_meta['log_conc'] = bool(prep['log_conc'])
    prep_meta['zscore'] = bool(prep['zscore'])
    if len(xs) < 3 or len(set(xs)) < 3:
        return jsonify({'error': '预处理后有效标定点不足（至少 3 个不同浓度），请调整预处理或补充数据'}), 400
    results = modeling.fit_all_models(xs, ys)
    # best 仅从可保存/可反解的主模型中选取（SVR/随机森林为对比模型不可保存）
    best = modeling.best_model({k: r for k, r in results.items() if not r.get('compare')})
    xmin, xmax = min(xs), max(xs)
    for mt, r in results.items():
        if 'error' not in r and not r.get('compare'):
            r['curve'] = modeling.curve_points(mt, r['params'], xmin, xmax)
        elif r.get('compare') and 'params' in r:
            r['params'] = {k: v for k, v in r['params'].items() if not hasattr(v, 'predict')}
    return jsonify({
        'ok': True,
        'feature': feature,
        'unit': (groups[0].get('unit') if groups else None) or 'ng/mL',
        'n': len(xs),
        'results': results,
        'best': best[0] if best else None,
        'data': [[round(x, 4), round(y, 4)] for x, y in zip(xs, ys)],
        'preprocess': prep_meta,
    })


# ---- 问题2-H：数据预处理设置与交叉验证 ----

def _preprocess_settings():
    raw = db.get_setting(_db_path(), 'modeling_preprocess') or '{}'
    try:
        d = json.loads(raw)
    except Exception:  # noqa: BLE001
        d = {}
    return {'log_conc': bool(d.get('log_conc')), 'zscore': bool(d.get('zscore')),
            'iqr': bool(d.get('iqr', True))}


@bp.route('/modeling/preprocess', methods=['GET', 'POST'])
def modeling_preprocess():
    """查看/设置建模预处理参数（log 浓度、Z-score 标准化、IQR 剔除）。"""
    if request.method == 'GET':
        return jsonify(_preprocess_settings())
    body = request.get_json(silent=True) or {}
    d = {'log_conc': bool(body.get('log_conc')),
         'zscore': bool(body.get('zscore')),
         'iqr': bool(body.get('iqr', True))}
    db.set_setting(_db_path(), 'modeling_preprocess', json.dumps(d, ensure_ascii=False))
    return jsonify({'ok': True, **d})


@bp.route('/modeling/explore', methods=['POST'])
def modeling_explore():
    """数据探索（问题2-I）：散点数据、浓度组分箱线数据、特征-浓度 Pearson 相关。"""
    body = request.get_json(silent=True) or {}
    feature = body.get('feature', 'hue')
    if feature not in modeling.FEATURES:
        imported = db.query_one(_db_path(),
                                "SELECT 1 FROM calibration_points WHERE source='import' AND feature_name=? LIMIT 1",
                                (feature,))
        if not imported:
            return jsonify({'error': f'不支持的特征：{feature}'}), 400
    xs, ys, _ = _collect_calibration_data(feature)
    if len(xs) < 3:
        return jsonify({'error': '有效标定点不足'}), 400
    prep = _preprocess_settings()
    xs_p, ys_p, _ = modeling.preprocess_data(xs, ys, log_conc=prep['log_conc'],
                                             zscore=prep['zscore'], iqr=prep['iqr'])
    pearson = 0.0
    try:
        import numpy as _np
        pearson = round(float(_np.corrcoef(xs_p, ys_p)[0, 1]), 4) if len(xs_p) > 1 else 0.0
    except Exception:  # noqa: BLE001
        pearson = 0.0
    by_conc = {}
    for xi, yi in zip(xs, ys):
        by_conc.setdefault(round(float(xi), 4), []).append(round(float(yi), 4))
    box = [{'conc': c, 'values': v} for c, v in sorted(by_conc.items())]
    return jsonify({'ok': True, 'feature': feature,
                    'points': [[round(x, 4), round(y, 4)] for x, y in zip(xs, ys)],
                    'box': box, 'pearson': pearson, 'n': len(xs)})


@bp.route('/modeling/cv', methods=['POST'])
def modeling_cv():
    """交叉验证（问题2-H）：loo / kfold / leave_group，对全部模型输出 R²/RMSE/MAE。"""
    body = request.get_json(silent=True) or {}
    feature = body.get('feature', 'hue')
    if feature not in modeling.FEATURES:
        imported = db.query_one(_db_path(),
                                "SELECT 1 FROM calibration_points WHERE source='import' AND feature_name=? LIMIT 1",
                                (feature,))
        if not imported:
            return jsonify({'error': f'不支持的特征：{feature}'}), 400
    method = body.get('method', 'loo')
    if method not in ('loo', 'kfold', 'leave_group'):
        return jsonify({'error': 'method 仅支持 loo / kfold / leave_group'}), 400
    try:
        k = max(2, int(body.get('k', 5)))
    except (TypeError, ValueError):
        k = 5
    xs, ys, _ = _collect_calibration_data(feature)
    if len(xs) < 3 or len(set(xs)) < 3:
        return jsonify({'error': '有效标定点不足（至少 3 个不同浓度）'}), 400
    prep = _preprocess_settings()
    xs, ys, prep_meta = modeling.preprocess_data(xs, ys, log_conc=prep['log_conc'],
                                                 zscore=prep['zscore'], iqr=prep['iqr'])
    prep_meta['log_conc'] = bool(prep['log_conc'])
    prep_meta['zscore'] = bool(prep['zscore'])
    out = {}
    for mt in modeling.MODEL_TYPES + modeling.COMPARE_MODEL_TYPES:
        try:
            out[mt] = modeling.cross_validate(xs, ys, mt, method, k)
        except ValueError as e:
            out[mt] = {'error': str(e)}
    return jsonify({'ok': True, 'results': out, 'preprocess': prep_meta,
                    'n': len(xs), 'method': method, 'k': k})


@bp.route('/models', methods=['GET'])
def list_models():
    rows = db.query(_db_path(), 'SELECT * FROM models ORDER BY is_active DESC, id DESC')
    return jsonify({'models': rows})


@bp.route('/models', methods=['POST'])
def create_model():
    """保存模型（通常来自 fit 结果）。body: {name, type, params, metrics, source_snapshot}"""
    body = request.get_json(silent=True) or {}
    mtype = body.get('type')
    if mtype not in modeling.MODEL_TYPES:
        return jsonify({'error': f'未知模型类型：{mtype}'}), 400
    params = body.get('params')
    if not isinstance(params, dict):
        return jsonify({'error': 'params 必须为对象'}), 400
    name = (body.get('name') or '').strip() or f'{mtype}-{json.dumps(params, ensure_ascii=False)[:20]}'
    metrics = body.get('metrics') or {}
    snapshot = body.get('source_snapshot') or {}
    mid = db.execute(_db_path(),
                     'INSERT INTO models (name, type, params_json, metrics_json, source_snapshot_json) '
                     'VALUES (?,?,?,?,?)',
                     (name, mtype, json.dumps(params, ensure_ascii=False),
                      json.dumps(metrics, ensure_ascii=False), json.dumps(snapshot, ensure_ascii=False)))
    # 首个模型自动激活
    if not db.query_one(_db_path(), 'SELECT id FROM models WHERE is_active=1 LIMIT 1'):
        db.execute(_db_path(), 'UPDATE models SET is_active=1 WHERE id=?', (mid,))
    return jsonify({'ok': True, 'id': mid})


@bp.route('/models/<int:mid>/activate', methods=['POST'])
def activate_model(mid):
    row = db.query_one(_db_path(), 'SELECT * FROM models WHERE id=?', (mid,))
    if not row:
        abort(404)
    db.execute(_db_path(), 'UPDATE models SET is_active=0')
    db.execute(_db_path(), 'UPDATE models SET is_active=1 WHERE id=?', (mid,))
    return jsonify({'ok': True, 'id': mid})


@bp.route('/models/<int:mid>', methods=['DELETE', 'POST'])
def delete_model(mid):
    row = db.query_one(_db_path(), 'SELECT * FROM models WHERE id=?', (mid,))
    if not row:
        abort(404)
    was_active = row['is_active']
    db.execute(_db_path(), 'DELETE FROM models WHERE id=?', (mid,))
    if was_active:
        rows = db.query(_db_path(), 'SELECT id FROM models ORDER BY id DESC LIMIT 1')
        if rows:
            db.execute(_db_path(), 'UPDATE models SET is_active=1 WHERE id=?', (rows[0]['id'],))
    return jsonify({'ok': True})


@bp.route('/models/<int:mid>/delete', methods=['POST'])
def delete_model_alias(mid):
    return delete_model(mid)


@bp.route('/models/<int:mid>/export', methods=['GET'])
def export_model_file(mid):
    """导出模型为 .joblib 文件（问题2-K）。

    文件内含 {name, type, params, metrics, source_snapshot}，可在另一台电脑导入复用。
    """
    row = db.query_one(_db_path(), 'SELECT * FROM models WHERE id=?', (mid,))
    if not row:
        abort(404)
    try:
        import joblib
    except ImportError:
        return jsonify({'error': '缺少 joblib，请执行 pip install joblib'}), 500
    payload = {
        'name': row['name'],
        'type': row['type'],
        'params': json.loads(row['params_json'] or '{}'),
        'metrics': json.loads(row['metrics_json'] or '{}'),
        'source_snapshot': json.loads(row['source_snapshot_json'] or '{}'),
    }
    buf = io.BytesIO()
    joblib.dump(payload, buf)
    buf.seek(0)
    fname = row['name'].replace('/', '_').replace('\\', '_') + '.joblib'
    return send_file(buf, as_attachment=True, download_name=fname, mimetype='application/octet-stream')


@bp.route('/models/import_joblib', methods=['POST'])
def import_model_file():
    """导入 .joblib 模型文件（问题2-K）。"""
    up = request.files.get('file')
    if not up or not up.filename:
        return jsonify({'error': '请选择 .joblib 模型文件'}), 400
    try:
        import joblib
        payload = joblib.load(up.stream)
    except Exception as e:  # noqa: BLE001
        return jsonify({'error': f'模型文件解析失败：{e}'}), 400
    mtype = payload.get('type')
    if mtype not in modeling.MODEL_TYPES:
        return jsonify({'error': f'文件中的模型类型不可作为生效模型：{mtype}'}), 400
    if not isinstance(payload.get('params'), dict):
        return jsonify({'error': '文件缺少 params'}), 400
    name = (payload.get('name') or '').strip() or f'{mtype}-import'
    mid = db.execute(_db_path(),
                     'INSERT INTO models (name, type, params_json, metrics_json, source_snapshot_json) '
                     'VALUES (?,?,?,?,?)',
                     (name, mtype, json.dumps(payload['params'], ensure_ascii=False),
                      json.dumps(payload.get('metrics') or {}, ensure_ascii=False),
                      json.dumps(payload.get('source_snapshot') or {}, ensure_ascii=False)))
    if not db.query_one(_db_path(), 'SELECT id FROM models WHERE is_active=1 LIMIT 1'):
        db.execute(_db_path(), 'UPDATE models SET is_active=1 WHERE id=?', (mid,))
    return jsonify({'ok': True, 'id': mid, 'name': name, 'type': mtype})


def _predict_from_value(fval, model):
    """按生效模型快照反解浓度 C±U（与 run_detection 同口径）。"""
    params = json.loads(model['params_json'] or '{}')
    snap = json.loads(model['source_snapshot_json'] or '{}')
    data = snap.get('data') or []
    if len(data) < 3:
        raise ValueError('生效模型缺少标定数据快照，无法计算不确定度；请重新标定')
    xs = [float(d[0]) for d in data]
    ys = [float(d[1]) for d in data]
    prep = snap.get('preprocess') or {}
    fval_d = float(fval)
    if prep.get('y_std'):
        fval_d = (fval_d - prep['y_mean']) / prep['y_std']
    res = modeling.predict_with_u(model['type'], params, fval_d, xs, ys)
    if prep.get('log_conc'):
        import math
        res['conc'] = max(10.0 ** res['conc'] - 1.0, 0.0)
        res['u'] = res['u'] * math.log(10.0) * (res['conc'] + 1.0)
    return res, snap


@bp.route('/modeling/predict_csv', methods=['POST'])
def predict_csv():
    """特征 CSV 批量预测（问题2-K）：CSV 需含生效模型的特征列，逐行输出 C±U。"""
    model = db.query_one(_db_path(), 'SELECT * FROM models WHERE is_active=1 ORDER BY id DESC LIMIT 1')
    if not model:
        return jsonify({'error': '尚未保存生效标定模型'}), 400
    up = request.files.get('file')
    if not up or not up.filename:
        return jsonify({'error': '请选择特征 CSV 文件'}), 400
    if not up.filename.lower().endswith('.csv'):
        return jsonify({'error': '仅支持 .csv'}), 400
    try:
        text = up.stream.read().decode('utf-8-sig')
        rows = list(csv.DictReader(io.StringIO(text)))
    except Exception as e:  # noqa: BLE001
        return jsonify({'error': f'CSV 解析失败：{e}'}), 400
    if not rows:
        return jsonify({'error': 'CSV 无数据行'}), 400
    snap = json.loads(model['source_snapshot_json'] or '{}')
    feature = snap.get('feature', 'hue')
    if not rows[0].get(feature):
        return jsonify({'error': f'CSV 缺少特征列“{feature}”（模型使用该特征）'}), 400
    lower = _num_setting('limit_lower')
    upper = _num_setting('limit_upper')
    out = []
    for i, r in enumerate(rows):
        try:
            val = float(r[feature])
            res, _ = _predict_from_value(val, model)
            status = _judge(res['conc'], res['u'], lower, upper)
            out.append({'row': i + 1, 'feature_value': round(val, 6),
                        'conc': res['conc'], 'u': res['u'], 'status': status})
        except (ValueError, KeyError) as e:
            out.append({'row': i + 1, 'feature_value': r.get(feature), 'error': str(e)})
    return jsonify({'ok': True, 'feature': feature, 'model': model['name'], 'predictions': out})


@bp.route('/modeling/predict_csv/export', methods=['POST'])
def predict_csv_export():
    """特征 CSV 批量预测并直接下载结果 CSV（问题2-K）。"""
    model = db.query_one(_db_path(), 'SELECT * FROM models WHERE is_active=1 ORDER BY id DESC LIMIT 1')
    if not model:
        return jsonify({'error': '尚未保存生效标定模型'}), 400
    up = request.files.get('file')
    if not up:
        return jsonify({'error': '请选择特征 CSV 文件'}), 400
    text = up.stream.read().decode('utf-8-sig')
    reader = list(csv.DictReader(io.StringIO(text)))
    if not reader:
        return jsonify({'error': 'CSV 无数据行'}), 400
    snap = json.loads(model['source_snapshot_json'] or '{}')
    feature = snap.get('feature', 'hue')
    lower = _num_setting('limit_lower')
    upper = _num_setting('limit_upper')
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['row', 'feature_value', 'predicted_conc', 'uncertainty_u', 'status'])
    for i, r in enumerate(reader):
        try:
            val = float(r[feature])
            res, _ = _predict_from_value(val, model)
            status = _judge(res['conc'], res['u'], lower, upper)
            w.writerow([i + 1, round(val, 6), res['conc'], res['u'], status])
        except (ValueError, KeyError) as e:
            w.writerow([i + 1, r.get(feature, ''), '', '', 'error: ' + str(e)])
    data = buf.getvalue().encode('utf-8-sig')
    return send_file(io.BytesIO(data), as_attachment=True, download_name='predictions.csv',
                     mimetype='text/csv')


# ---------------- M6：浓度检测与超限判定 ----------------

def _judge(conc, u, lower, upper):
    """超限判定：C±U 与上下限比较。返回 within|above|below|borderline。"""
    if lower is not None and conc + u < lower:
        return 'below'
    if upper is not None and conc - u > upper:
        return 'above'
    if lower is not None and conc - u < lower:
        return 'borderline'
    if upper is not None and conc + u > upper:
        return 'borderline'
    return 'within'


def run_detection(image_id):
    """对检测图执行浓度检测：反解 C、计算 U（95% 预测区间）、超限判定、写记录与快照。

    无图抛 KeyError；无特征/无生效模型等抛 ValueError。
    """
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        raise KeyError(image_id)
    if row['kind'] != 'detection':
        raise ValueError('仅检测图可执行浓度检测（标定图请到建模页加入分组）')
    feat = db.query_one(_db_path(), 'SELECT * FROM features WHERE image_id=?', (image_id,))
    if not feat:
        raise ValueError('尚无特征，请先完成 ROI 与特征提取')
    model = db.query_one(_db_path(), 'SELECT * FROM models WHERE is_active=1 ORDER BY id DESC LIMIT 1')
    if not model:
        raise ValueError('尚未保存生效标定模型，请先到建模页标定并保存')
    params = json.loads(model['params_json'] or '{}')
    metrics = json.loads(model['metrics_json'] or '{}')
    snap = json.loads(model['source_snapshot_json'] or '{}')
    feature = snap.get('feature', 'hue')
    fval = feat.get(feature) if feat else None
    if fval is None:
        # 问题2-G/H：组合/自定义特征（如 T_R_over_Bg_R、deltaE_T_vs_Bg）从 roi_features 派生
        try:
            from ..image_processing import derive_combined_features
            rf_rows = db.query(_db_path(), 'SELECT * FROM roi_features WHERE image_id=?', (image_id,))
            feats_map = {}
            for r in rf_rows:
                try:
                    feats_map[r['roi_name']] = json.loads(r['features_json'])
                except Exception:  # noqa: BLE001
                    continue
            comb = derive_combined_features(feats_map)
            fval = comb.get(feature)
            if fval is None and feature in feats_map.get('T', {}):
                fval = feats_map['T'].get(feature)
        except Exception:  # noqa: BLE001
            fval = None
    if fval is None:
        raise ValueError(f'该图缺少特征“{feature}”，无法检测')
    data = snap.get('data') or []
    if len(data) < 3:
        raise ValueError('生效模型缺少标定数据快照，无法计算不确定度；请重新标定')
    xs = [float(d[0]) for d in data]
    ys = [float(d[1]) for d in data]

    # 问题2-H：应用保存模型时的预处理（Z-score 特征 → 预测后反标准化；log 浓度 → 反解）
    prep = snap.get('preprocess') or {}
    fval_d = float(fval)
    if prep.get('y_std'):
        fval_d = (fval_d - prep['y_mean']) / prep['y_std']
    res = modeling.predict_with_u(model['type'], params, fval_d, xs, ys)
    if prep.get('log_conc'):
        import math
        res['conc'] = max(10.0 ** res['conc'] - 1.0, 0.0)
        res['u'] = res['u'] * math.log(10.0) * (res['conc'] + 1.0)
    lower = _num_setting('limit_lower')
    upper = _num_setting('limit_upper')
    status = _judge(res['conc'], res['u'], lower, upper)

    unit = (snap.get('unit') or '').strip() or 'ng/mL'
    full_snapshot = {
        'model_id': model['id'],
        'model_name': model['name'],
        'model_type': model['type'],
        'model_params': params,
        'model_metrics': metrics,
        'feature': feature,
        'feature_value': round(float(fval), 6),
        'calibration_n': len(xs),
        'calibration_data': [[round(a, 6), round(b, 6)] for a, b in zip(xs, ys)],
        'limits': {'lower': lower, 'upper': upper},
        'conf': 0.95,
    }
    existing = db.query_one(_db_path(), 'SELECT id FROM detections WHERE image_id=?', (image_id,))
    if existing:
        db.execute(_db_path(),
                   'UPDATE detections SET model_id=?, conc=?, u=?, status=?, unit=?, '
                   'params_snapshot_json=?, batch=?, created_at=datetime(\'now\',\'localtime\') WHERE image_id=?',
                   (model['id'], res['conc'], res['u'], status, unit,
                    json.dumps(full_snapshot, ensure_ascii=False), row['batch'], image_id))
        det_id = existing['id']
    else:
        det_id = db.execute(_db_path(),
                            'INSERT INTO detections (image_id, model_id, conc, u, status, unit, params_snapshot_json, batch) '
                            'VALUES (?,?,?,?,?,?,?,?)',
                            (image_id, model['id'], res['conc'], res['u'], status, unit,
                             json.dumps(full_snapshot, ensure_ascii=False), row['batch']))
    _upsert_step(image_id, 'result', json.dumps({'conc': res['conc'], 'u': res['u'], 'status': status},
                                                ensure_ascii=False), 'ok')
    return {
        'id': det_id, 'image_id': image_id, 'conc': res['conc'], 'u': res['u'], 'unit': unit,
        'status': status, 'feature': feature, 'feature_value': fval,
        'model_name': model['name'], 'model_r2': metrics.get('r2'),
        'limits': {'lower': lower, 'upper': upper},
    }


def _num_setting(key):
    raw = db.get_setting(_db_path(), key)
    if raw is None or raw == '':
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _upsert_step(image_id, step, params_json, status, error=None):
    """与 api_workflow 共用的步骤快照写入。"""
    from .api_workflow import _upsert_step as _wf_upsert
    return _wf_upsert(image_id, step, params_json, status, error)


@bp.route('/detect/<int:image_id>', methods=['POST'])
def detect_image(image_id):
    try:
        result = run_detection(image_id)
    except KeyError:
        abort(404)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'detection': result})


@bp.route('/detections', methods=['GET'])
def list_detections():
    """按时间、浓度、批次检索检测记录（M7 记录页使用）。"""
    q = request.args.get('q', '').strip()
    batch = request.args.get('batch', '').strip()
    where = []
    params = []
    if q:
        where.append('(i.batch LIKE ? OR d.status LIKE ?)')
        params += [f'%{q}%', f'%{q}%']
    if batch:
        where.append('i.batch = ?')
        params.append(batch)
    sql = ('SELECT d.*, i.batch AS image_batch, i.created_at AS image_created, i.file_path '
           'FROM detections d JOIN images i ON i.id = d.image_id ')
    if where:
        sql += 'WHERE ' + ' AND '.join(where)
    sql += ' ORDER BY d.created_at DESC'
    rows = db.query(_db_path(), sql, params)
    return jsonify({'detections': rows})


# ---------------- M7：首页聚合与记录导出 ----------------

@bp.route('/home/summary', methods=['GET'])
def home_summary():
    data_dir = current_app.config['DATA_DIR']
    model = db.query_one(_db_path(), 'SELECT * FROM models WHERE is_active=1 ORDER BY id DESC LIMIT 1')
    metrics = json.loads(model['metrics_json']) if model and model['metrics_json'] else {}
    groups = db.query(_db_path(), 'SELECT COUNT(*) AS n FROM calibration_groups')
    points = db.query_one(_db_path(),
                          'SELECT COUNT(*) AS n FROM calibration_points cp JOIN calibration_groups cg '
                          'ON cg.id=cp.group_id WHERE cp.included=1')
    last = db.query_one(_db_path(),
                        'SELECT d.*, m.name AS model_name FROM detections d '
                        'LEFT JOIN models m ON m.id=d.model_id '
                        'ORDER BY d.id DESC LIMIT 1')
    today = db.query_one(_db_path(),
                         "SELECT COUNT(*) AS n FROM images WHERE date(created_at)=date('now','localtime')")
    total_det = db.query_one(_db_path(), 'SELECT COUNT(*) AS n FROM detections')
    batch = db.get_setting(_db_path(), 'current_batch')

    # 问题1：最近检测图的工作流每步图像（原图/预处理/通道分离/ROI 叠加）
    dd = Path(data_dir)
    wf_steps = None
    if last:
        iid = last['image_id']
        wf_steps = {'image_id': iid, 'upload': f'/api/images/{iid}/original'}
        if (dd / 'processed' / f'{iid}.png').exists():
            wf_steps['preprocess'] = f'/api/images/{iid}/processed'
        ch = {}
        for ch_name in 'rgb':
            if (dd / 'processed' / 'channels' / f'{iid}_{ch_name}.png').exists():
                ch[ch_name] = f'/api/images/{iid}/channel/{ch_name}'
        if ch:
            wf_steps['channels'] = ch
        wf_steps['roi'] = f'/api/images/{iid}/overlay'

    return jsonify({
        'model_name': model['name'] if model else None,
        'model_type': model['type'] if model else None,
        'model_r2': metrics.get('r2'),
        'model_lod': metrics.get('lod'),
        'calibrated': bool(groups and groups[0]['n'] > 0),
        'n_groups': (groups or [{}])[0]['n'] if groups else 0,
        'n_points': points['n'] if points else 0,
        'last_detection': {
            'image_id': last['image_id'], 'conc': last['conc'], 'u': last['u'],
            'unit': last.get('unit') or 'ng/mL',
            'status': last['status'], 'model_name': last['model_name'],
            'created_at': last['created_at'],
        } if last else None,
        'workflow_steps': wf_steps,
        'today_count': today['n'] if today else 0,
        'total_detections': total_det['n'] if total_det else 0,
        'current_batch': batch,
        'data_dir': str(data_dir),
    })


@bp.route('/detections/export', methods=['GET'])
def export_detections():
    """导出检测记录为 CSV。"""
    import csv
    import io as _io

    q = request.args.get('q', '').strip()
    batch = request.args.get('batch', '').strip()
    where, params = [], []
    if q:
        where.append('(i.batch LIKE ? OR d.status LIKE ?)')
        params += [f'%{q}%', f'%{q}%']
    if batch:
        where.append('i.batch = ?')
        params.append(batch)
    sql = ('SELECT d.*, i.batch AS image_batch FROM detections d JOIN images i ON i.id = d.image_id ')
    if where:
        sql += 'WHERE ' + ' AND '.join(where)
    sql += ' ORDER BY d.created_at DESC'
    rows = db.query(_db_path(), sql, params)
    buf = _io.StringIO()
    w = csv.writer(buf)
    w.writerow(['ID', '图片ID', '批次', '浓度C', '不确定度U', '判定', '检测时间'])
    for r in rows:
        w.writerow([r['id'], r['image_id'], r['image_batch'], r['conc'], r['u'],
                    r['status'], r['created_at']])
    resp = current_app.response_class(buf.getvalue(), mimetype='text/csv; charset=utf-8')
    resp.headers['Content-Disposition'] = 'attachment; filename=detections.csv'
    return resp


