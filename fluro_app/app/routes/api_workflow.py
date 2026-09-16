# -*- coding: utf-8 -*-
"""工作流相关 API：上传（M1）→ 预处理（M2）→ ROI/特征（M3）→ 流水线/结果（M4+）。"""
import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request, send_file, abort

from .. import database as db
from ..image_processing import (read_image, make_thumbnail, preprocess,
                                extract_features, auto_roi,
                                roi_to_pixels, crop_roi)

bp = Blueprint('workflow', __name__, url_prefix='/api')


def _db_path():
    return Path(current_app.config['DATA_DIR']) / 'fluro.db'


def _data_dir():
    return Path(current_app.config['DATA_DIR'])


def _img_dir(kind='original'):
    base = _data_dir()
    if kind == 'thumb':
        return base / 'thumbnails'
    if kind == 'processed':
        return base / 'processed'
    if kind == 'ref':
        return base / 'refs'
    return base / 'images'


def _upsert_step(image_id, step, params_json, status, error=None):
    existing = db.query_one(_db_path(),
                            'SELECT id FROM pipeline_steps WHERE image_id=? AND step=?',
                            (image_id, step))
    if existing:
        db.execute(_db_path(),
                   'UPDATE pipeline_steps SET params_json=?, status=?, error=?, '
                   'updated_at=datetime(\'now\',\'localtime\') WHERE id=?',
                   (params_json, status, error, existing['id']))
        return existing['id']
    return db.execute(_db_path(),
                      'INSERT INTO pipeline_steps (image_id, step, params_json, status, error) '
                      'VALUES (?,?,?,?,?)',
                      (image_id, step, params_json, status, error))


def _save_uploaded(file_storage):
    """保存上传文件：原图 + 缩略图，返回 (abs_path, thumb_path)。"""
    ext = Path(file_storage.filename).suffix.lower() or '.png'
    if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'):
        ext = '.png'
    date_dir = datetime.now().strftime('%Y%m%d')
    odir = _img_dir('original') / date_dir
    tdir = _img_dir('thumb')
    odir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex[:12]
    raw_path = odir / f'{uid}{ext}'
    file_storage.save(str(raw_path))

    img = read_image(raw_path)
    if img is None:
        raw_path.unlink(missing_ok=True)
        raise ValueError(f'无法解码图片：{file_storage.filename}')
    thumb = make_thumbnail(img)
    thumb_path = tdir / f'{uid}.jpg'
    cv2_thumb = thumb[:, :, ::-1]  # RGB -> BGR 供 cv2 保存
    import cv2
    cv2.imwrite(str(thumb_path), cv2_thumb)
    return str(raw_path), str(thumb_path)


@bp.route('/images/upload', methods=['POST'])
def upload_images():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': '未收到文件'}), 400
    kind = request.form.get('kind', 'detection')
    if kind not in ('calibration', 'detection'):
        kind = 'detection'
    batch = (request.form.get('batch') or '').strip() or None
    conc_raw = (request.form.get('known_conc') or '').strip()
    known_conc = None
    if conc_raw:
        try:
            known_conc = float(conc_raw)
        except ValueError:
            return jsonify({'error': f'已知浓度格式错误：{conc_raw}'}), 400
    if kind == 'calibration' and known_conc is None:
        return jsonify({'error': '标定图必须填写已知浓度'}), 400

    if batch:
        db.execute(_db_path(), 'INSERT OR IGNORE INTO batches (name) VALUES (?)', (batch,))

    results = []
    errors = []
    for f in files:
        try:
            raw_path, thumb_path = _save_uploaded(f)
        except ValueError as e:
            errors.append(str(e))
            continue
        img_id = db.execute(
            _db_path(),
            'INSERT INTO images (file_path, thumb_path, kind, batch, known_conc) VALUES (?,?,?,?,?)',
            (raw_path, thumb_path, kind, batch, known_conc),
        )
        results.append({
            'id': img_id,
            'kind': kind,
            'batch': batch,
            'known_conc': known_conc,
            'thumb_url': f'/api/images/{img_id}/thumb',
        })
    return jsonify({'ok': True, 'images': results, 'errors': errors})


@bp.route('/images/<int:image_id>/thumb')
def image_thumb(image_id):
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row or not row.get('thumb_path'):
        abort(404)
    return send_file(row['thumb_path'], mimetype='image/jpeg')


@bp.route('/images/<int:image_id>/original')
def image_original(image_id):
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    return send_file(row['file_path'])


def _png_response(img_rgb):
    """RGB ndarray → PNG 响应。"""
    ok, buf = cv2.imencode('.png', img_rgb[:, :, ::-1])
    if not ok:
        abort(500)
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/png')


@bp.route('/images/<int:image_id>/overlay')
def image_overlay(image_id):
    """处理后图（无则原图）叠加 ROI 框，返回 PNG。用于各步骤的图片展示。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    img = _source_image(row)
    if img is None:
        abort(404)
    roi_row = db.query_one(_db_path(), 'SELECT * FROM roi WHERE image_id=?', (image_id,))
    if roi_row:
        try:
            x0, y0, x1, y1 = roi_to_pixels((roi_row['x'], roi_row['y'], roi_row['w'], roi_row['h']), img.shape)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 3)
        except ValueError:
            pass
    return _png_response(img)


@bp.route('/images/<int:image_id>/roi_crop')
def image_roi_crop(image_id):
    """ROI 区域裁剪图（处理后图优先）。无 ROI 返回 404。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    img = _source_image(row)
    if img is None:
        abort(404)
    roi_row = db.query_one(_db_path(), 'SELECT * FROM roi WHERE image_id=?', (image_id,))
    if not roi_row:
        abort(404)
    try:
        crop = crop_roi(img, (roi_row['x'], roi_row['y'], roi_row['w'], roi_row['h']))
    except ValueError:
        abort(404)
    return _png_response(crop)


@bp.route('/images/<int:image_id>/roi_avg')
def image_roi_avg(image_id):
    """ROI 区域平均色块图（128x128 纯色 PNG）。无 ROI 返回 404。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    img = _source_image(row)
    if img is None:
        abort(404)
    roi_row = db.query_one(_db_path(), 'SELECT * FROM roi WHERE image_id=?', (image_id,))
    if not roi_row:
        abort(404)
    try:
        crop = crop_roi(img, (roi_row['x'], roi_row['y'], roi_row['w'], roi_row['h']))
    except ValueError:
        abort(404)
    mean = crop.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    block = np.full((128, 128, 3), mean, dtype=np.uint8)
    return _png_response(block)


@bp.route('/images/<int:image_id>', methods=['GET'])
def image_info(image_id):
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    return jsonify(row)


@bp.route('/images/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    """删除一张图：清理相关表（特征/ROI/流水线步骤/检测记录/标定点）与磁盘文件。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    for tbl in ('features', 'roi', 'pipeline_steps', 'detections', 'calibration_points'):
        db.execute(_db_path(), f'DELETE FROM {tbl} WHERE image_id=?', (image_id,))
    db.execute(_db_path(), 'DELETE FROM images WHERE id=?', (image_id,))
    for p in (row.get('file_path'), row.get('thumb_path'),
              _img_dir('processed') / f'{image_id}.png'):
        if p:
            Path(p).unlink(missing_ok=True)
    return jsonify({'ok': True})


@bp.route('/images', methods=['GET'])
def list_images():
    rows = db.query(_db_path(), 'SELECT * FROM images ORDER BY id DESC')
    for r in rows:
        r['thumb_url'] = f"/api/images/{r['id']}/thumb"
    return jsonify({'images': rows})


# ---------------- M4：流水线 ----------------

@bp.route('/pipeline/run', methods=['POST'])
def run_pipeline_batch():
    """批量执行自动流水线。body: {image_ids: [...]}；缺省处理全部 uploaded 图。"""
    from ..pipeline import run_batch as pipeline_run_batch
    body = request.get_json(silent=True) or {}
    ids = body.get('image_ids')
    results = pipeline_run_batch(ids)
    return jsonify({'ok': True, 'results': results})


@bp.route('/pipeline/<int:image_id>/run', methods=['POST'])
def run_pipeline_single(image_id):
    from ..pipeline import run_single as pipeline_run_single
    result = pipeline_run_single(image_id)
    return jsonify({'ok': True, 'result': result})


# ---------------- M2：预处理 ----------------

@bp.route('/refs', methods=['GET'])
def get_refs():
    return jsonify({
        'dark_path': db.get_setting(_db_path(), 'dark_ref_path'),
        'flat_path': db.get_setting(_db_path(), 'flat_ref_path'),
    })


@bp.route('/refs/<string:kind>', methods=['POST'])
def upload_ref(kind):
    """上传暗场/平场参考图并保存到 settings。kind: dark | flat"""
    if kind not in ('dark', 'flat'):
        return jsonify({'error': 'kind 必须为 dark 或 flat'}), 400
    f = request.files.get('file')
    if not f:
        return jsonify({'error': '未收到文件'}), 400
    rdir = _img_dir('ref')
    rdir.mkdir(parents=True, exist_ok=True)
    ext = Path(f.filename).suffix.lower() or '.png'
    path = rdir / f'{kind}_ref{ext}'
    f.save(str(path))
    img = read_image(path)
    if img is None:
        path.unlink(missing_ok=True)
        return jsonify({'error': '参考图无法解码'}), 400
    db.set_setting(_db_path(), f'{kind}_ref_path', str(path))
    return jsonify({'ok': True, 'kind': kind, 'path': str(path)})


@bp.route('/images/<int:image_id>/processed')
def image_processed(image_id):
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    path = _img_dir('processed') / f'{image_id}.png'
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype='image/png')


@bp.route('/pipeline/<int:image_id>/preprocess', methods=['POST'])
def preprocess_image(image_id):
    """执行预处理（可调参数重跑）。"""
    body = request.get_json(silent=True) or {}
    method = body.get('filter', current_app.config['DEFAULT_FILTER'])
    kernel = body.get('kernel', current_app.config['DEFAULT_KERNEL'])
    use_dark = bool(body.get('use_dark', False))
    use_flat = bool(body.get('use_flat', False))
    dark_path = body.get('dark_path')
    flat_path = body.get('flat_path')
    try:
        return jsonify({'ok': True, **preprocess_core(image_id, method, kernel, use_dark, use_flat, dark_path, flat_path)})
    except KeyError:
        abort(404)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


def preprocess_core(image_id, method=None, kernel=None, use_dark=False, use_flat=False,
                    dark_path=None, flat_path=None):
    """预处理核心：读取原图 → 校正/去噪 → 保存处理图 → 写快照。无图抛 KeyError。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        raise KeyError(image_id)
    if method not in ('gaussian', 'median'):
        method = current_app.config['DEFAULT_FILTER']
    try:
        kernel = int(kernel) if kernel is not None else current_app.config['DEFAULT_KERNEL']
    except (TypeError, ValueError):
        kernel = current_app.config['DEFAULT_KERNEL']

    dark_path = dark_path or db.get_setting(_db_path(), 'dark_ref_path')
    flat_path = flat_path or db.get_setting(_db_path(), 'flat_ref_path')

    img = read_image(row['file_path'])
    if img is None:
        raise ValueError('原图无法读取')
    dark = read_image(dark_path) if (use_dark and dark_path) else None
    flat = read_image(flat_path) if (use_flat and flat_path) else None
    if use_dark and dark_path and dark is None:
        raise ValueError(f'暗场参考图无法读取：{dark_path}')
    if use_flat and flat_path and flat is None:
        raise ValueError(f'平场参考图无法读取：{flat_path}')

    out = preprocess(img, method=method, kernel=kernel, dark=dark, flat=flat)
    pdir = _img_dir('processed')
    pdir.mkdir(parents=True, exist_ok=True)
    out_path = pdir / f'{image_id}.png'
    cv2.imwrite(str(out_path), out[:, :, ::-1])

    params = {'filter': method, 'kernel': kernel, 'use_dark': use_dark, 'use_flat': use_flat}
    _upsert_step(image_id, 'preprocess', json.dumps(params, ensure_ascii=False), 'ok')

    # 级联：预处理变更后重算下游（特征；检测结果在 M6 追加）
    from .api_model import recompute_downstream
    recompute_downstream(image_id)

    return {'image_id': image_id, 'params': params, 'processed_url': f'/api/images/{image_id}/processed'}


# ---------------- M3：ROI 与特征 ----------------

def _source_image(row):
    """特征/ROI 计算优先用处理后图，其次原图。"""
    processed = _img_dir('processed') / f"{row['id']}.png"
    if processed.exists():
        img = read_image(processed)
        if img is not None:
            return img
    return read_image(row['file_path'])


def _active_template():
    return db.query_one(_db_path(),
                        'SELECT * FROM roi_templates WHERE is_active=1 ORDER BY id DESC LIMIT 1')


@bp.route('/templates', methods=['GET'])
def list_templates():
    rows = db.query(_db_path(), 'SELECT * FROM roi_templates ORDER BY id DESC')
    return jsonify({'templates': rows})


@bp.route('/templates', methods=['POST'])
def create_template():
    body = request.get_json(silent=True) or {}
    name = (body.get('name') or '').strip() or '模板'
    try:
        x, y, w, h = (float(body[k]) for k in ('x', 'y', 'w', 'h'))
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'x/y/w/h 必须为数字'}), 400
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1 and y + h <= 1):
        return jsonify({'error': 'ROI 超出图像范围'}), 400
    ref_image_id = body.get('ref_image_id')
    tid = db.execute(_db_path(),
                     'INSERT INTO roi_templates (name, x, y, w, h, ref_image_id) VALUES (?,?,?,?,?,?)',
                     (name, x, y, w, h, ref_image_id))
    if _active_template() is None:
        db.execute(_db_path(), 'UPDATE roi_templates SET is_active=1 WHERE id=?', (tid,))
    return jsonify({'ok': True, 'id': tid, 'is_active': True if _active_template() and _active_template()['id'] == tid else False})


@bp.route('/templates/<int:tid>', methods=['DELETE', 'POST'])
def delete_template(tid):
    db.execute(_db_path(), 'DELETE FROM roi_templates WHERE id=?', (tid,))
    if _active_template() is None:
        rows = db.query(_db_path(), 'SELECT * FROM roi_templates ORDER BY id DESC LIMIT 1')
        if rows:
            db.execute(_db_path(), 'UPDATE roi_templates SET is_active=1 WHERE id=?', (rows[0]['id'],))
    return jsonify({'ok': True})


@bp.route('/templates/<int:tid>/delete', methods=['POST'])
def delete_template_alias(tid):
    return delete_template(tid)


@bp.route('/templates/<int:tid>/activate', methods=['POST'])
def activate_template(tid):
    row = db.query_one(_db_path(), 'SELECT * FROM roi_templates WHERE id=?', (tid,))
    if not row:
        abort(404)
    db.execute(_db_path(), 'UPDATE roi_templates SET is_active=0')
    db.execute(_db_path(), 'UPDATE roi_templates SET is_active=1 WHERE id=?', (tid,))
    return jsonify({'ok': True, 'id': tid})


@bp.route('/pipeline/<int:image_id>/roi', methods=['GET'])
def get_roi(image_id):
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    roi = db.query_one(_db_path(), 'SELECT * FROM roi WHERE image_id=?', (image_id,))
    return jsonify({'roi': roi})


@bp.route('/pipeline/<int:image_id>/roi', methods=['POST'])
def save_roi(image_id):
    """保存手动修正后的 ROI（级联：重算特征与结果）。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    body = request.get_json(silent=True) or {}
    try:
        x, y, w, h = (float(body[k]) for k in ('x', 'y', 'w', 'h'))
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'x/y/w/h 必须为数字'}), 400
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1 and y + h <= 1):
        return jsonify({'error': 'ROI 超出图像范围'}), 400
    source = body.get('source', 'manual')
    if source not in ('auto', 'manual'):
        source = 'manual'

    existing = db.query_one(_db_path(), 'SELECT id FROM roi WHERE image_id=?', (image_id,))
    if existing:
        db.execute(_db_path(),
                   'UPDATE roi SET x=?, y=?, w=?, h=?, source=?, updated_at=datetime(\'now\',\'localtime\') WHERE id=?',
                   (x, y, w, h, source, existing['id']))
    else:
        db.execute(_db_path(),
                   'INSERT INTO roi (image_id, x, y, w, h, source) VALUES (?,?,?,?,?,?)',
                   (image_id, x, y, w, h, source))
    _upsert_step(image_id, 'roi', json.dumps({'x': x, 'y': y, 'w': w, 'h': h, 'source': source}, ensure_ascii=False), 'ok')

    # 级联：ROI 变更后重算特征与结果
    from .api_model import recompute_downstream
    recompute_downstream(image_id)

    return jsonify({'ok': True, 'roi': {'x': x, 'y': y, 'w': w, 'h': h, 'source': source}})


@bp.route('/pipeline/<int:image_id>/roi/auto', methods=['POST'])
def auto_roi_image(image_id):
    """用激活模板自动套用（含模板匹配微调）并保存。"""
    try:
        roi = auto_roi_core(image_id)
    except KeyError:
        abort(404)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'roi': roi})


def auto_roi_core(image_id):
    """ROI 自动套用核心。无图抛 KeyError；无模板抛 ValueError。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        raise KeyError(image_id)
    tpl = _active_template()
    if not tpl:
        raise ValueError('尚未创建 ROI 模板，请先在参考图上框选并保存为模板')
    img = _source_image(row)
    if img is None:
        raise ValueError('图像无法读取')
    ref_img = None
    if tpl.get('ref_image_id'):
        ref_row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (tpl['ref_image_id'],))
        if ref_row:
            ref_img = read_image(ref_row['file_path'])
    roi = auto_roi(img, (tpl['x'], tpl['y'], tpl['w'], tpl['h']), ref_img_rgb=ref_img)

    existing = db.query_one(_db_path(), 'SELECT id FROM roi WHERE image_id=?', (image_id,))
    if existing:
        db.execute(_db_path(),
                   'UPDATE roi SET x=?, y=?, w=?, h=?, source=?, updated_at=datetime(\'now\',\'localtime\') WHERE id=?',
                   (roi[0], roi[1], roi[2], roi[3], 'auto', existing['id']))
    else:
        db.execute(_db_path(),
                   'INSERT INTO roi (image_id, x, y, w, h, source) VALUES (?,?,?,?,?,?)',
                   (image_id, roi[0], roi[1], roi[2], roi[3], 'auto'))
    _upsert_step(image_id, 'roi', json.dumps({'x': roi[0], 'y': roi[1], 'w': roi[2], 'h': roi[3], 'source': 'auto'}, ensure_ascii=False), 'ok')

    from .api_model import recompute_downstream
    recompute_downstream(image_id)
    return {'x': roi[0], 'y': roi[1], 'w': roi[2], 'h': roi[3], 'source': 'auto'}


def compute_features_core(image_id):
    """计算并保存检测区特征（优先处理后图）。无图抛 KeyError；无 ROI 或空 ROI 抛 ValueError。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        raise KeyError(image_id)
    roi_row = db.query_one(_db_path(), 'SELECT * FROM roi WHERE image_id=?', (image_id,))
    if not roi_row:
        raise ValueError('请先设置 ROI（自动套用或手动框选）')
    img = _source_image(row)
    if img is None:
        raise ValueError('图像无法读取')
    feats = extract_features(img, (roi_row['x'], roi_row['y'], roi_row['w'], roi_row['h']))

    db.execute(_db_path(),
               'INSERT INTO features (image_id, mean_r, mean_g, mean_b, hue, saturation, value, '
               'ratio_gr, ratio_bg, intensity, texture_entropy) VALUES (?,?,?,?,?,?,?,?,?,?,?) '
               'ON CONFLICT(image_id) DO UPDATE SET mean_r=excluded.mean_r, mean_g=excluded.mean_g, '
               'mean_b=excluded.mean_b, hue=excluded.hue, saturation=excluded.saturation, '
               'value=excluded.value, ratio_gr=excluded.ratio_gr, ratio_bg=excluded.ratio_bg, '
               'intensity=excluded.intensity, texture_entropy=excluded.texture_entropy, '
               'updated_at=datetime(\'now\',\'localtime\')',
               (image_id, feats['mean_r'], feats['mean_g'], feats['mean_b'], feats['hue'],
                feats['saturation'], feats['value'], feats['ratio_gr'], feats['ratio_bg'],
                feats['intensity'], feats['texture_entropy']))
    _upsert_step(image_id, 'feature', json.dumps(feats, ensure_ascii=False), 'ok')
    return feats


@bp.route('/pipeline/<int:image_id>/features', methods=['POST'])
def compute_features(image_id):
    try:
        feats = compute_features_core(image_id)
    except KeyError:
        abort(404)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'features': feats})


@bp.route('/pipeline/<int:image_id>/features', methods=['GET'])
def get_features(image_id):
    row = db.query_one(_db_path(), 'SELECT * FROM features WHERE image_id=?', (image_id,))
    if not row:
        return jsonify({'features': None})
    return jsonify({'features': row})
