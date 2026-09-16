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
from flask import Blueprint, Response, current_app, jsonify, request, send_file, abort

from .. import database as db
from ..image_processing import (read_image, make_thumbnail, preprocess,
                                extract_features, auto_roi,
                                roi_to_pixels, crop_roi,
                                auto_detect_roi, bg_ring_mean, apply_bg_subtraction,
                                extract_roi_features, derive_combined_features,
                                white_balance_correct)

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
    """保存上传文件：原图 + 缩略图，返回 (abs_path, thumb_path, filename)。"""
    filename = Path(file_storage.filename).name or f'图片_{uuid.uuid4().hex[:8]}.png'
    ext = Path(filename).suffix.lower() or '.png'
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
        raise ValueError(f'无法解码图片：{filename}')
    thumb = make_thumbnail(img)
    thumb_path = tdir / f'{uid}.jpg'
    cv2_thumb = thumb[:, :, ::-1]  # RGB -> BGR 供 cv2 保存
    import cv2
    cv2.imwrite(str(thumb_path), cv2_thumb)
    return str(raw_path), str(thumb_path), filename


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
            raw_path, thumb_path, filename = _save_uploaded(f)
        except ValueError as e:
            errors.append(str(e))
            continue
        img_id = db.execute(
            _db_path(),
            'INSERT INTO images (file_path, thumb_path, kind, batch, known_conc, filename) VALUES (?,?,?,?,?,?)',
            (raw_path, thumb_path, kind, batch, known_conc, filename),
        )
        results.append({
            'id': img_id,
            'kind': kind,
            'batch': batch,
            'known_conc': known_conc,
            'filename': filename,
            'thumb_url': f'/api/images/{img_id}/thumb',
        })
    return jsonify({'ok': True, 'images': results, 'errors': errors})


@bp.route('/images/<int:image_id>/thumb')
def image_thumb(image_id):
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row or not row.get('thumb_path'):
        abort(404)
    return send_file(row['thumb_path'], mimetype='image/jpeg')


@bp.route('/images/<int:image_id>', methods=['PATCH'])
def update_image_meta(image_id):
    """编辑图片元信息（问题2-E）：批次 / 已知浓度 / 重复编号 / 备注。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    body = request.get_json(silent=True) or {}
    sets, args = [], []
    if 'batch' in body:
        batch = (body.get('batch') or '').strip() or None
        sets.append('batch=?')
        args.append(batch)
        if batch:
            db.execute(_db_path(), 'INSERT OR IGNORE INTO batches (name) VALUES (?)', (batch,))
    if 'known_conc' in body:
        v = body.get('known_conc')
        if v in (None, ''):
            sets.append('known_conc=NULL')
        else:
            try:
                v = float(v)
            except (TypeError, ValueError):
                return jsonify({'error': '已知浓度格式错误'}), 400
            sets.append('known_conc=?')
            args.append(v)
    if 'note' in body:
        sets.append('note=?')
        args.append((body.get('note') or '').strip() or None)
    if 'replicate' in body:
        rep_raw = body.get('replicate')
        try:
            rep = int(rep_raw) if rep_raw not in (None, '') else 1
        except (TypeError, ValueError):
            return jsonify({'error': '重复编号必须为整数'}), 400
        if rep < 1:
            return jsonify({'error': '重复编号必须 ≥ 1'}), 400
        sets.append('replicate=?')
        args.append(rep)
    if not sets:
        return jsonify({'ok': True})
    args.append(image_id)
    db.execute(_db_path(), f"UPDATE images SET {', '.join(sets)} WHERE id=?", args)
    return jsonify({'ok': True})


# ---- 问题2-E：特征数据导出（CSV / Excel，单卡片规格列） ----

EXPORT_FEATURE_COLUMNS = [
    'image_name', 'image_path', 'batch', 'concentration', 'replicate',
    'T_R', 'T_G', 'T_B', 'T_H', 'T_S', 'T_V', 'T1_L', 'T1_a', 'T1_b',
    'Bg_R', 'Bg_G', 'Bg_B',
    'deltaE_T_vs_Bg',
    'T_R_over_Bg_R', 'T_G_over_Bg_G', 'T_B_over_Bg_B',
    'OD_T_R', 'OD_T_G', 'OD_T_B',
    'note',
]


def _export_feature_rows():
    """按单卡片规格组装每图一行（有 ROI 特征的图）。"""
    imgs = db.query(_db_path(), 'SELECT * FROM images ORDER BY id')
    rows = []
    for im in imgs:
        rf = db.query(_db_path(), 'SELECT * FROM roi_features WHERE image_id=?', (im['id'],))
        if not rf:
            continue
        feats = {}
        for r in rf:
            try:
                feats[r['roi_name']] = json.loads(r['features_json'])
            except Exception:  # noqa: BLE001
                continue
        if not feats:
            continue
        t = feats.get('T') or next((v for k, v in feats.items()
                                    if k not in ('Bg', 'background', 'Blank', 'blank')), None)
        bg = feats.get('Bg') or feats.get('background') or feats.get('Blank')
        comb = derive_combined_features(feats)
        row = {
            'image_name': im.get('filename') or f"image_{im['id']}",
            'image_path': im['file_path'],
            'batch': im.get('batch') or '',
            'concentration': im.get('known_conc') if im.get('known_conc') is not None else '',
            'replicate': im.get('replicate') or 1,
            'T_R': t.get('mean_r') if t else '', 'T_G': t.get('mean_g') if t else '',
            'T_B': t.get('mean_b') if t else '', 'T_H': t.get('hue') if t else '',
            'T_S': t.get('saturation') if t else '', 'T_V': t.get('value') if t else '',
            'T1_L': t.get('lab_l') if t else '', 'T1_a': t.get('lab_a') if t else '',
            'T1_b': t.get('lab_b') if t else '',
            'Bg_R': bg.get('mean_r') if bg else '', 'Bg_G': bg.get('mean_g') if bg else '',
            'Bg_B': bg.get('mean_b') if bg else '',
            'deltaE_T_vs_Bg': comb.get('deltaE_T_vs_Bg', ''),
            'T_R_over_Bg_R': comb.get('T_R_over_Bg_R', ''),
            'T_G_over_Bg_G': comb.get('T_G_over_Bg_G', ''),
            'T_B_over_Bg_B': comb.get('T_B_over_Bg_B', ''),
            'OD_T_R': t.get('od_r') if t else '', 'OD_T_G': t.get('od_g') if t else '',
            'OD_T_B': t.get('od_b') if t else '',
            'note': im.get('note') or '',
        }
        rows.append(row)
    return rows


@bp.route('/export/features.csv')
def export_features_csv():
    import csv as _csv
    import io as _io
    rows = _export_feature_rows()
    buf = _io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=EXPORT_FEATURE_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    resp = Response(buf.getvalue(), mimetype='text/csv; charset=utf-8')
    resp.headers['Content-Disposition'] = 'attachment; filename=features.csv'
    return resp


@bp.route('/export/features.xlsx')
def export_features_xlsx():
    import io as _io
    try:
        from openpyxl import Workbook
    except ImportError:
        return jsonify({'error': '未安装 openpyxl，请运行 pip install openpyxl'}), 500
    rows = _export_feature_rows()
    wb = Workbook()
    ws = wb.active
    ws.title = 'features'
    ws.append(EXPORT_FEATURE_COLUMNS)
    for r in rows:
        ws.append([r.get(c, '') for c in EXPORT_FEATURE_COLUMNS])
    buf = _io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    resp = Response(buf.getvalue(),
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    resp.headers['Content-Disposition'] = 'attachment; filename=features.xlsx'
    return resp


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
    """处理后图（无则原图）叠加全部命名 ROI 框与名称，返回 PNG。用于各步骤的图片展示。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    img = _source_image(row)
    if img is None:
        abort(404)
    rois = db.query(_db_path(), 'SELECT * FROM rois WHERE image_id=? ORDER BY id', (image_id,))
    if rois:
        colors = [(0, 255, 255), (255, 200, 0), (0, 200, 255), (200, 0, 255), (255, 0, 200), (0, 255, 200)]
        for i, r in enumerate(rois):
            try:
                x0, y0, x1, y1 = roi_to_pixels((r['x'], r['y'], r['w'], r['h']), img.shape)
            except ValueError:
                continue
            col = colors[i % len(colors)]
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 3)
            cv2.putText(img, str(r['name']), (x0, max(14, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    else:
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
    """主检测区（T）裁剪图（处理后图优先）。无 ROI 返回 404。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    img = _source_image(row)
    if img is None:
        abort(404)
    main = _main_roi_rect(image_id)
    if not main:
        abort(404)
    try:
        crop = crop_roi(img, (main[1], main[2], main[3], main[4]))
    except ValueError:
        abort(404)
    return _png_response(crop)


@bp.route('/images/<int:image_id>/roi_avg')
def image_roi_avg(image_id):
    """主检测区（T）平均色块图（128x128 纯色 PNG）。无 ROI 返回 404。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    img = _source_image(row)
    if img is None:
        abort(404)
    main = _main_roi_rect(image_id)
    if not main:
        abort(404)
    try:
        crop = crop_roi(img, (main[1], main[2], main[3], main[4]))
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
    for tbl in ('features', 'roi_features', 'roi', 'rois', 'pipeline_steps', 'detections', 'calibration_points'):
        db.execute(_db_path(), f'DELETE FROM {tbl} WHERE image_id=?', (image_id,))
    db.execute(_db_path(), 'DELETE FROM images WHERE id=?', (image_id,))
    for p in (row.get('file_path'), row.get('thumb_path'),
              _img_dir('processed') / f'{image_id}.png'):
        if p:
            Path(p).unlink(missing_ok=True)
    cdir = _data_dir() / 'processed' / 'channels'
    for ch in 'rgb':
        (cdir / f'{image_id}_{ch}.png').unlink(missing_ok=True)
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


# ---------------- 问题5：RGB 通道分离 ----------------

def channels_core(image_id):
    """RGB 通道分离核心：处理后图分离 R/G/B 灰度图并保存。无图抛 KeyError。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        raise KeyError(image_id)
    img = _source_image(row)
    if img is None:
        raise ValueError('图像无法读取')
    cdir = _data_dir() / 'processed' / 'channels'
    cdir.mkdir(parents=True, exist_ok=True)
    info = {}
    for i, ch in enumerate('rgb'):
        chan = img[:, :, i]
        ok, buf = cv2.imencode('.png', chan)
        if not ok:
            raise ValueError('通道图编码失败')
        (cdir / f'{image_id}_{ch}.png').write_bytes(buf.tobytes())
        info[f'{ch}_url'] = f'/api/images/{image_id}/channel/{ch}'
        info[f'mean_{ch}'] = round(float(chan.mean()), 3)
    _upsert_step(image_id, 'channels', json.dumps(info, ensure_ascii=False), 'ok')
    return info


@bp.route('/pipeline/<int:image_id>/channels', methods=['POST'])
def compute_channels(image_id):
    try:
        info = channels_core(image_id)
    except KeyError:
        abort(404)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'channels': info})


@bp.route('/pipeline/<int:image_id>/channels', methods=['GET'])
def get_channels(image_id):
    step = db.query_one(_db_path(),
                        'SELECT * FROM pipeline_steps WHERE image_id=? AND step=?',
                        (image_id, 'channels'))
    if not step:
        return jsonify({'channels': None})
    info = json.loads(step['params_json'] or '{}')
    return jsonify({'channels': info})


@bp.route('/images/<int:image_id>/channel/<string:ch>', methods=['GET'])
def channel_image(image_id, ch):
    if ch not in ('r', 'g', 'b'):
        return jsonify({'error': 'ch 必须为 r/g/b'}), 400
    p = _data_dir() / 'processed' / 'channels' / f'{image_id}_{ch}.png'
    if not p.exists():
        abort(404)
    return send_file(str(p))


# ---------------- M2：预处理 ----------------

@bp.route('/refs', methods=['GET'])
def get_refs():
    return jsonify({
        'dark_path': db.get_setting(_db_path(), 'dark_ref_path'),
        'flat_path': db.get_setting(_db_path(), 'flat_ref_path'),
    })


@bp.route('/refs/<string:kind>/image', methods=['GET'])
def ref_image(kind):
    """返回暗场/平场参考图（PNG/原格式）。未设置时 404。"""
    if kind not in ('dark', 'flat'):
        return jsonify({'error': 'kind 必须为 dark 或 flat'}), 400
    path = db.get_setting(_db_path(), f'{kind}_ref_path')
    if not path or not Path(path).exists():
        abort(404)
    return send_file(path)


@bp.route('/refs/<string:kind>', methods=['DELETE'])
def clear_ref(kind):
    """清除指定参考图（删除文件并清空设置）。"""
    if kind not in ('dark', 'flat'):
        return jsonify({'error': 'kind 必须为 dark 或 flat'}), 400
    path = db.get_setting(_db_path(), f'{kind}_ref_path')
    if path:
        Path(path).unlink(missing_ok=True)
    db.set_setting(_db_path(), f'{kind}_ref_path', None)
    return jsonify({'ok': True, 'kind': kind})


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
    use_wb = bool(body.get('use_wb', False))
    wb_roi_name = body.get('wb_roi_name')
    dark_path = body.get('dark_path')
    flat_path = body.get('flat_path')
    try:
        return jsonify({'ok': True, **preprocess_core(image_id, method, kernel, use_dark, use_flat,
                                                      dark_path, flat_path, use_wb, wb_roi_name)})
    except KeyError:
        abort(404)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


def preprocess_core(image_id, method=None, kernel=None, use_dark=False, use_flat=False,
                    dark_path=None, flat_path=None, use_wb=False, wb_roi_name=None):
    """预处理核心：读取原图 → 校正/去噪 → 保存处理图 → 写快照。无图抛 KeyError。

    use_wb + wb_roi_name：以指定命名 ROI（如 Bg 背景区）的平均色为白参考做白平衡增益校正。
    """
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
    wb_used = False
    if use_wb:
        wb_roi = None
        if wb_roi_name:
            wb_roi = db.query_one(_db_path(),
                                  'SELECT * FROM rois WHERE image_id=? AND name=?',
                                  (image_id, wb_roi_name))
        if not wb_roi:
            wb_roi = db.query_one(_db_path(),
                                  "SELECT * FROM rois WHERE image_id=? AND role='background' "
                                  "ORDER BY id LIMIT 1", (image_id,))
        if not wb_roi:
            main = _main_roi_rect(image_id)
            if main:
                wb_roi = {'x': main[1], 'y': main[2], 'w': main[3], 'h': main[4]}
        if wb_roi:
            try:
                white = crop_roi(out, (wb_roi['x'], wb_roi['y'], wb_roi['w'], wb_roi['h']))
                white_mean = white.reshape(-1, 3).mean(axis=0)
                out = white_balance_correct(out, white_mean)
                wb_used = True
            except ValueError:
                raise ValueError('白参考 ROI 无效，请检查 ROI 设置')
        else:
            raise ValueError('开启白平衡校正但未找到白参考 ROI：请先设置背景区（Bg）ROI')

    pdir = _img_dir('processed')
    pdir.mkdir(parents=True, exist_ok=True)
    out_path = pdir / f'{image_id}.png'
    cv2.imwrite(str(out_path), out[:, :, ::-1])

    params = {'filter': method, 'kernel': kernel, 'use_dark': use_dark, 'use_flat': use_flat,
              'use_wb': use_wb, 'wb_roi_name': wb_roi_name, 'wb_used': wb_used}
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


# ---- 问题2-A：多命名 ROI（单卡片 T/Bg）----

def _main_roi_rect(image_id):
    """主检测区：优先 rois 表 role='sample'（或首个），回退旧 roi 表（兼容迁移前数据）。"""
    rows = db.query(_db_path(), 'SELECT * FROM rois WHERE image_id=? ORDER BY id', (image_id,))
    if rows:
        for r in rows:
            if r['role'] == 'sample':
                return (r['name'], r['x'], r['y'], r['w'], r['h'], r.get('bg_subtract') or 0)
        r0 = rows[0]
        return (r0['name'], r0['x'], r0['y'], r0['w'], r0['h'], r0.get('bg_subtract') or 0)
    old = db.query_one(_db_path(), 'SELECT * FROM roi WHERE image_id=?', (image_id,))
    if old:
        return ('T', old['x'], old['y'], old['w'], old['h'], old.get('bg_subtract') or 0)
    return None


def _validate_roi_items(items):
    """校验 ROI 列表，返回 [(name, role, x, y, w, h, bg_subtract)]，非法抛 ValueError。"""
    if not isinstance(items, list) or not items:
        raise ValueError('rois 不能为空')
    seen, clean = set(), []
    for it in items:
        try:
            name = str(it['name']).strip()
            x, y, w, h = (float(it[k]) for k in ('x', 'y', 'w', 'h'))
        except (KeyError, TypeError, ValueError):
            raise ValueError('每个 ROI 需 name 与数字 x/y/w/h')
        if not name:
            raise ValueError('ROI 名称不能为空')
        if name in seen:
            raise ValueError(f'ROI 名称重复：{name}')
        seen.add(name)
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1 and y + h <= 1):
            raise ValueError(f'ROI {name} 超出图像范围')
        role = str(it.get('role') or 'sample').strip() or 'sample'
        if role not in ('sample', 'background', 'color_card', 'blank'):
            role = 'sample'
        clean.append((name, role, x, y, w, h, 1 if it.get('bg_subtract') else 0))
    return clean


def _write_rois(image_id, clean, source='manual', recompute=True):
    """全量写入多 ROI（事务替换），并把主检测区（role=sample 首行）同步到旧 roi 表保持兼容。"""
    db.execute(_db_path(), 'DELETE FROM rois WHERE image_id=?', (image_id,))
    for c in clean:
        db.execute(_db_path(),
                   'INSERT INTO rois (image_id, name, role, x, y, w, h, source, bg_subtract) '
                   'VALUES (?,?,?,?,?,?,?,?,?)',
                   (image_id, c[0], c[1], c[2], c[3], c[4], c[5], source, c[6]))
    main = next((c for c in clean if c[1] == 'sample'), clean[0])
    existing = db.query_one(_db_path(), 'SELECT * FROM roi WHERE image_id=?', (image_id,))
    if existing:
        db.execute(_db_path(),
                   'UPDATE roi SET x=?, y=?, w=?, h=?, source=?, bg_subtract=?, '
                   'updated_at=datetime(\'now\',\'localtime\') WHERE id=?',
                   (main[2], main[3], main[4], main[5], source, main[6], existing['id']))
    else:
        db.execute(_db_path(),
                   'INSERT INTO roi (image_id, x, y, w, h, source, bg_subtract) VALUES (?,?,?,?,?,?,?)',
                   (image_id, main[2], main[3], main[4], main[5], source, main[6]))
    _upsert_step(image_id, 'roi',
                 json.dumps([{'name': c[0], 'role': c[1], 'x': c[2], 'y': c[3], 'w': c[4], 'h': c[5],
                              'bg_subtract': c[6], 'source': source} for c in clean],
                            ensure_ascii=False), 'ok')
    if recompute:
        from .api_model import recompute_downstream
        recompute_downstream(image_id)


@bp.route('/images/<int:image_id>/rois', methods=['GET'])
def get_rois(image_id):
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    rois = db.query(_db_path(), 'SELECT * FROM rois WHERE image_id=? ORDER BY id', (image_id,))
    if not rois:
        old = db.query_one(_db_path(), 'SELECT * FROM roi WHERE image_id=?', (image_id,))
        if old:
            rois = [{'name': 'T', 'role': 'sample', 'x': old['x'], 'y': old['y'],
                     'w': old['w'], 'h': old['h'], 'source': old['source'],
                     'bg_subtract': old.get('bg_subtract') or 0}]
    return jsonify({'rois': rois})


@bp.route('/images/<int:image_id>/rois', methods=['POST'])
def save_rois(image_id):
    """全量保存多命名 ROI（T/Bg 等）。body: {rois: [{name, role, x,y,w,h, bg_subtract}]}"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    body = request.get_json(silent=True) or {}
    try:
        clean = _validate_roi_items(body.get('rois'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    _write_rois(image_id, clean, source='manual')
    return jsonify({'ok': True, 'rois': clean})


@bp.route('/images/<int:image_id>/rois/apply_template', methods=['POST'])
def apply_roi_template(image_id):
    """套用 ROI 模板（优先多 ROI 集合 template_json；否则单 ROI 作为主检测区 T）。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    body = request.get_json(silent=True) or {}
    tid = body.get('template_id')
    tpl = db.query_one(_db_path(), 'SELECT * FROM roi_templates WHERE id=?', (tid,)) if tid else _active_template()
    if not tpl:
        abort(404)
    tj = tpl.get('template_json')
    if tj:
        try:
            parsed = json.loads(tj)
        except Exception:  # noqa: BLE001
            parsed = None
        if parsed:
            items = [dict(r, name=n) for n, r in parsed.items()]
            try:
                clean = _validate_roi_items(items)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
            _write_rois(image_id, clean, source='template')
            return jsonify({'ok': True, 'rois': clean})
    items = [{'name': 'T', 'role': 'sample', 'x': tpl['x'], 'y': tpl['y'],
              'w': tpl['w'], 'h': tpl['h'], 'bg_subtract': 0}]
    _write_rois(image_id, _validate_roi_items(items), source='template')
    return jsonify({'ok': True, 'rois': items})


@bp.route('/templates', methods=['GET'])
def list_templates():
    rows = db.query(_db_path(), 'SELECT * FROM roi_templates ORDER BY id DESC')
    return jsonify({'templates': rows})


@bp.route('/templates', methods=['POST'])
def create_template():
    body = request.get_json(silent=True) or {}
    name = (body.get('name') or '').strip() or '模板'
    template_json = body.get('template_json')
    if template_json:
        # 多 ROI 集合模板：{"T": {"x","y","w","h","role"}, "Bg": {...}}
        if not isinstance(template_json, dict) or not template_json:
            return jsonify({'error': 'template_json 必须为非空对象'}), 400
        clean = {}
        for rname, r in template_json.items():
            rname = str(rname).strip()
            if not rname:
                return jsonify({'error': 'ROI 名称不能为空'}), 400
            try:
                x, y, w, h = (float(r[k]) for k in ('x', 'y', 'w', 'h'))
            except (KeyError, TypeError, ValueError):
                return jsonify({'error': f'ROI {rname} 的 x/y/w/h 必须为数字'}), 400
            if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1 and y + h <= 1):
                return jsonify({'error': f'ROI {rname} 超出图像范围'}), 400
            clean[rname] = {
                'x': x, 'y': y, 'w': w, 'h': h,
                'role': str(r.get('role') or 'sample'),
                'bg_subtract': 1 if r.get('bg_subtract') else 0,
            }
        tid = db.execute(_db_path(),
                         'INSERT INTO roi_templates (name, template_json) VALUES (?,?)',
                         (name, json.dumps(clean, ensure_ascii=False)))
    else:
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
    bg_subtract = 1 if body.get('bg_subtract') else 0

    existing = db.query_one(_db_path(), 'SELECT * FROM roi WHERE image_id=?', (image_id,))
    if existing:
        db.execute(_db_path(),
                   'UPDATE roi SET x=?, y=?, w=?, h=?, source=?, bg_subtract=?, updated_at=datetime(\'now\',\'localtime\') WHERE id=?',
                   (x, y, w, h, source, bg_subtract, existing['id']))
    else:
        db.execute(_db_path(),
                   'INSERT INTO roi (image_id, x, y, w, h, source, bg_subtract) VALUES (?,?,?,?,?,?,?)',
                   (image_id, x, y, w, h, source, bg_subtract))
    _upsert_step(image_id, 'roi', json.dumps({'x': x, 'y': y, 'w': w, 'h': h, 'source': source, 'bg_subtract': bg_subtract}, ensure_ascii=False), 'ok')

    # 同步多 ROI 表主检测区（单 ROI 编辑语义 → role='sample'）
    main_row = db.query_one(_db_path(),
                            "SELECT id FROM rois WHERE image_id=? AND role='sample'", (image_id,))
    if main_row:
        db.execute(_db_path(),
                   'UPDATE rois SET x=?, y=?, w=?, h=?, source=?, bg_subtract=? WHERE id=?',
                   (x, y, w, h, source, bg_subtract, main_row['id']))
    else:
        db.execute(_db_path(),
                   'INSERT INTO rois (image_id, name, role, x, y, w, h, source, bg_subtract) '
                   'VALUES (?,?,?,?,?,?,?,?,?)',
                   (image_id, 'T', 'sample', x, y, w, h, source, bg_subtract))

    # 级联：ROI 变更后重算特征与结果
    from .api_model import recompute_downstream
    recompute_downstream(image_id)

    return jsonify({'ok': True, 'roi': {'x': x, 'y': y, 'w': w, 'h': h, 'source': source, 'bg_subtract': bg_subtract}})


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
    """ROI 自动套用核心。无图抛 KeyError；找不到检测区抛 ValueError。

    有激活模板 → 模板映射/匹配微调；无模板 → 基于图像内容的自动识别。
    """
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        raise KeyError(image_id)
    img = _source_image(row)
    if img is None:
        raise ValueError('图像无法读取')
    tpl = _active_template()
    if tpl and tpl.get('template_json'):
        # 多 ROI 集合模板（问题2）：T/Bg 等直接套用，无需图像匹配
        try:
            tj = json.loads(tpl['template_json'])
        except Exception as e:  # noqa: BLE001
            raise ValueError(f'模板数据解析失败：{e}')
        clean = []
        for rname, r in tj.items():
            role = str(r.get('role') or 'sample')
            clean.append((rname, role, float(r['x']), float(r['y']),
                          float(r['w']), float(r['h']), 1 if r.get('bg_subtract') else 0))
        _write_rois(image_id, clean, source='template', recompute=False)
        _upsert_step(image_id, 'roi', json.dumps({'source': 'template', 'n': len(clean)}, ensure_ascii=False), 'ok')
        return len(clean)
    if tpl:
        ref_img = None
        if tpl.get('ref_image_id'):
            ref_row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (tpl['ref_image_id'],))
            if ref_row:
                ref_img = read_image(ref_row['file_path'])
        roi = auto_roi(img, (tpl['x'], tpl['y'], tpl['w'], tpl['h']), ref_img_rgb=ref_img)
    else:
        roi = auto_detect_roi(img)
        if roi is None:
            raise ValueError('自动识别未找到明显检测区，请手动框选')

    existing = db.query_one(_db_path(), 'SELECT * FROM roi WHERE image_id=?', (image_id,))
    bg = existing['bg_subtract'] if existing else 0
    if existing:
        db.execute(_db_path(),
                   'UPDATE roi SET x=?, y=?, w=?, h=?, source=?, updated_at=datetime(\'now\',\'localtime\') WHERE id=?',
                   (roi[0], roi[1], roi[2], roi[3], 'auto', existing['id']))
    else:
        db.execute(_db_path(),
                   'INSERT INTO roi (image_id, x, y, w, h, source, bg_subtract) VALUES (?,?,?,?,?,?,?)',
                   (image_id, roi[0], roi[1], roi[2], roi[3], 'auto', 0))
    # 多 ROI 表同步：自动识别结果作为主检测区（实验配置首个 ROI 名或 T）
    try:
        exp = json.loads(db.get_setting(_db_path(), 'experiment') or '{}')
    except Exception:  # noqa: BLE001
        exp = {}
    main_name = (exp.get('roi_names') or ['T'])[0] if exp.get('roi_names') else 'T'
    clean = [(main_name, 'sample', roi[0], roi[1], roi[2], roi[3], bg)]
    _write_rois(image_id, clean, source='auto', recompute=False)

    _upsert_step(image_id, 'roi', json.dumps({'x': roi[0], 'y': roi[1], 'w': roi[2], 'h': roi[3], 'source': 'auto', 'bg_subtract': bg}, ensure_ascii=False), 'ok')

    from .api_model import recompute_downstream
    recompute_downstream(image_id)
    return {'x': roi[0], 'y': roi[1], 'w': roi[2], 'h': roi[3], 'source': 'auto', 'bg_subtract': bg}


def compute_features_core(image_id):
    """计算并保存检测区特征（优先处理后图）。无图抛 KeyError；无 ROI 或空 ROI 抛 ValueError。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        raise KeyError(image_id)
    main = _main_roi_rect(image_id)
    if not main:
        raise ValueError('请先设置 ROI（自动套用或手动框选）')
    img = _source_image(row)
    if img is None:
        raise ValueError('图像无法读取')
    roi_rect = (main[1], main[2], main[3], main[4])
    feats = extract_features(img, roi_rect)
    if main[5]:
        bg = bg_ring_mean(img, roi_rect)
        feats = apply_bg_subtraction(feats, bg)

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


# ---- 问题2-B：全 ROI 扩展特征 ----

def _save_legacy_features(image_id, feats):
    """主 ROI 兼容特征写入旧 features 表（检测/建模链路继续可用）。"""
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


def _all_rois_for(image_id):
    """读取该图全部 ROI（多 ROI 表优先，兼容旧单 ROI 表）。"""
    rois = db.query(_db_path(), 'SELECT * FROM rois WHERE image_id=? ORDER BY id', (image_id,))
    if rois:
        return rois
    old = db.query_one(_db_path(), 'SELECT * FROM roi WHERE image_id=?', (image_id,))
    if old:
        return [{'name': 'T', 'role': 'sample', 'x': old['x'], 'y': old['y'],
                 'w': old['w'], 'h': old['h'], 'bg_subtract': old.get('bg_subtract') or 0}]
    return []


def compute_all_roi_features_core(image_id):
    """计算全部命名 ROI 的扩展特征（问题2-B），存 roi_features 表并同步主 ROI 兼容特征。"""
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        raise KeyError(image_id)
    img = _source_image(row)
    if img is None:
        raise ValueError('图像无法读取')
    rois = _all_rois_for(image_id)
    if not rois:
        raise ValueError('请先设置 ROI（自动套用或手动框选）')
    feats_map = {}
    for r in rois:
        rect = (r['x'], r['y'], r['w'], r['h'])
        try:
            f = extract_roi_features(img, rect)
        except ValueError as e:
            raise ValueError(f'ROI {r["name"]} 无效：{e}')
        if r.get('bg_subtract'):
            try:
                f = apply_bg_subtraction(f, bg_ring_mean(img, rect))
            except ValueError:
                pass
        feats_map[r['name']] = f
        db.execute(_db_path(),
                   'INSERT INTO roi_features (image_id, roi_name, features_json) VALUES (?,?,?) '
                   'ON CONFLICT(image_id, roi_name) DO UPDATE SET features_json=excluded.features_json, '
                   'updated_at=datetime(\'now\',\'localtime\')',
                   (image_id, r['name'], json.dumps(f, ensure_ascii=False)))
    combined = derive_combined_features(feats_map)
    # 主 ROI 兼容特征写旧表
    main = next((r for r in rois if r['role'] == 'sample'), rois[0])
    legacy = extract_features(img, (main['x'], main['y'], main['w'], main['h']))
    if main.get('bg_subtract'):
        try:
            legacy = apply_bg_subtraction(legacy, bg_ring_mean(img, (main['x'], main['y'], main['w'], main['h'])))
        except ValueError:
            pass
    _save_legacy_features(image_id, legacy)
    _upsert_step(image_id, 'feature', json.dumps({'rois': feats_map, 'combined': combined},
                                                 ensure_ascii=False), 'ok')
    return {'rois': feats_map, 'combined': combined}


@bp.route('/images/<int:image_id>/features', methods=['POST'])
def compute_image_features(image_id):
    try:
        res = compute_all_roi_features_core(image_id)
    except KeyError:
        abort(404)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **res})


@bp.route('/images/<int:image_id>/features', methods=['GET'])
def get_image_features(image_id):
    row = db.query_one(_db_path(), 'SELECT * FROM images WHERE id=?', (image_id,))
    if not row:
        abort(404)
    rows = db.query(_db_path(), 'SELECT * FROM roi_features WHERE image_id=? ORDER BY roi_name',
                    (image_id,))
    feats_map = {}
    for r in rows:
        try:
            feats_map[r['roi_name']] = json.loads(r['features_json'])
        except Exception:  # noqa: BLE001
            continue
    combined = derive_combined_features(feats_map)
    return jsonify({'rois': feats_map, 'combined': combined})

