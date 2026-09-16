# -*- coding: utf-8 -*-
"""标定建模 API（M5）：浓度分组、数据点纳入/剔除、多模型拟合、模型库管理。"""
import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, abort

from .. import database as db
from .. import modeling

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
    """收集全部标定数据点：返回 (xs, ys, points)，points 含各分组信息。"""
    rows = db.query(_db_path(),
                    'SELECT cp.id AS point_id, cp.group_id, cp.included, cp.image_id, '
                    'cg.conc, cg.name AS group_name, f.* '
                    'FROM calibration_points cp '
                    'JOIN calibration_groups cg ON cg.id = cp.group_id '
                    'JOIN features f ON f.image_id = cp.image_id '
                    'ORDER BY cg.conc, cp.id')
    groups_map = {}
    xs, ys, points = [], [], []
    for r in rows:
        groups_map.setdefault(r['group_id'], {
            'id': r['group_id'], 'name': r['group_name'], 'conc': r['conc'], 'points': [],
        })
        feat = r.get(feature) if feature else r.get('hue')
        groups_map[r['group_id']]['points'].append({
            'point_id': r['point_id'], 'image_id': r['image_id'], 'included': r['included'],
            'features': {k: r[k] for k in modeling.FEATURES if k in r},
        })
        if feature:
            points.append(r)
            if r['included']:
                xs.append(r['conc'])
                ys.append(feat)
    groups = list(groups_map.values())
    return xs, ys, groups


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
        for k in modeling.FEATURES:
            vals = [p[k] for p in inc if p.get(k) is not None]
            if vals:
                means[k] = round(sum(vals) / len(vals), 4)
                sds[k] = round((sum((v - means[k]) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5, 4) if len(vals) > 1 else None
        g['mean'] = means
        g['sd'] = sds
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
    """用当前 included 数据点拟合 4 类模型。body: {feature: 'hue'}"""
    body = request.get_json(silent=True) or {}
    feature = body.get('feature', 'hue')
    if feature not in modeling.FEATURES:
        return jsonify({'error': f'不支持的特征：{feature}（可选：{", ".join(modeling.FEATURES)}）'}), 400
    xs, ys, groups = _collect_calibration_data(feature)
    if len(xs) < 3 or len(set(xs)) < 3:
        return jsonify({'error': '有效标定点不足（至少 3 个不同浓度、每浓度有特征数据）'}), 400
    results = modeling.fit_all_models(xs, ys)
    best = modeling.best_model(results)
    xmin, xmax = min(xs), max(xs)
    for mt, r in results.items():
        if 'error' not in r:
            r['curve'] = modeling.curve_points(mt, r['params'], xmin, xmax)
    return jsonify({
        'ok': True,
        'feature': feature,
        'n': len(xs),
        'results': results,
        'best': best[0] if best else None,
        'data': [[round(x, 4), round(y, 4)] for x, y in zip(xs, ys)],
    })


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
    fval = feat.get(feature)
    if fval is None:
        raise ValueError(f'该图缺少特征“{feature}”，无法检测')
    data = snap.get('data') or []
    if len(data) < 3:
        raise ValueError('生效模型缺少标定数据快照，无法计算不确定度；请重新标定')
    xs = [float(d[0]) for d in data]
    ys = [float(d[1]) for d in data]

    res = modeling.predict_with_u(model['type'], params, fval, xs, ys)
    lower = _num_setting('limit_lower')
    upper = _num_setting('limit_upper')
    status = _judge(res['conc'], res['u'], lower, upper)

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
                   'UPDATE detections SET model_id=?, conc=?, u=?, status=?, params_snapshot_json=?, '
                   'batch=?, created_at=datetime(\'now\',\'localtime\') WHERE image_id=?',
                   (model['id'], res['conc'], res['u'], status,
                    json.dumps(full_snapshot, ensure_ascii=False), row['batch'], image_id))
        det_id = existing['id']
    else:
        det_id = db.execute(_db_path(),
                            'INSERT INTO detections (image_id, model_id, conc, u, status, params_snapshot_json, batch) '
                            'VALUES (?,?,?,?,?,?,?)',
                            (image_id, model['id'], res['conc'], res['u'], status,
                             json.dumps(full_snapshot, ensure_ascii=False), row['batch']))
    _upsert_step(image_id, 'result', json.dumps({'conc': res['conc'], 'u': res['u'], 'status': status},
                                                ensure_ascii=False), 'ok')
    return {
        'id': det_id, 'image_id': image_id, 'conc': res['conc'], 'u': res['u'],
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
            'status': last['status'], 'model_name': last['model_name'],
            'created_at': last['created_at'],
        } if last else None,
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
