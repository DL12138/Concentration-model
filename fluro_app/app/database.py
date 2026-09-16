# -*- coding: utf-8 -*-
"""SQLite 数据层：版本化建表、通用读写。所有写操作事务安全。"""
import os
import sqlite3
from pathlib import Path

from .config import Config

# 版本化迁移：每个版本是一组 SQL。新增表/字段时追加 (version, [sql...])，不得修改旧版本。
MIGRATIONS = [
    (1, [
        """CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            thumb_path TEXT,
            kind TEXT NOT NULL DEFAULT 'detection',      -- calibration | detection
            batch TEXT,
            known_conc REAL,
            status TEXT NOT NULL DEFAULT 'uploaded',     -- uploaded|processing|ok|attention|error
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""",
        """CREATE TABLE IF NOT EXISTS pipeline_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            step TEXT NOT NULL,                          -- preprocess|roi|feature|result
            params_json TEXT,
            status TEXT NOT NULL DEFAULT 'pending',      -- pending|ok|error|manual
            error TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""",
        """CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL UNIQUE,
            mean_r REAL, mean_g REAL, mean_b REAL,
            hue REAL, saturation REAL, value REAL,
            ratio_gr REAL, ratio_bg REAL,
            intensity REAL, texture_entropy REAL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""",
        """CREATE TABLE IF NOT EXISTS roi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL UNIQUE,
            x REAL, y REAL, w REAL, h REAL,             -- 归一化 0~1
            source TEXT NOT NULL DEFAULT 'auto',         -- auto|manual
            updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""",
        """CREATE TABLE IF NOT EXISTS roi_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            x REAL, y REAL, w REAL, h REAL,             -- 归一化 0~1
            ref_image_id INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""",
        """CREATE TABLE IF NOT EXISTS calibration_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            conc REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""",
        """CREATE TABLE IF NOT EXISTS calibration_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER NOT NULL,
            image_id INTEGER NOT NULL,
            included INTEGER NOT NULL DEFAULT 1
        )""",
        """CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,                          -- linear|poly2|exp|4pl
            params_json TEXT,
            metrics_json TEXT,
            is_active INTEGER NOT NULL DEFAULT 0,
            source_snapshot_json TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""",
        """CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            model_id INTEGER,
            conc REAL, u REAL,
            status TEXT NOT NULL DEFAULT 'normal',       -- normal|over|under
            params_snapshot_json TEXT,
            batch TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""",
        """CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""",
    ]),
    (2, [
        """ALTER TABLE roi_templates ADD COLUMN is_active INTEGER NOT NULL DEFAULT 0""",
    ]),
    (3, [
        """ALTER TABLE roi ADD COLUMN bg_subtract INTEGER NOT NULL DEFAULT 0""",
    ]),
    (4, [
        # 单卡片槽比色检测：每张图多个命名 ROI（T 检测区、Bg 背景区等）
        """CREATE TABLE IF NOT EXISTS rois (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'sample',   -- sample|background|color_card|blank
            x REAL, y REAL, w REAL, h REAL,
            source TEXT NOT NULL DEFAULT 'manual', -- auto|manual|template
            bg_subtract INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            UNIQUE(image_id, name)
        )""",
        # 模板扩展：支持多 ROI 集合（JSON：{"T": {...}, "Bg": {...}}）
        """ALTER TABLE roi_templates ADD COLUMN template_json TEXT""",
    ]),
    (5, [
        # 每 ROI 扩展特征（问题2-B）：RGB 均值/中位数/SD、HSV、Lab、灰度、OD、通道比
        """CREATE TABLE IF NOT EXISTS roi_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            roi_name TEXT NOT NULL,
            features_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            UNIQUE(image_id, roi_name)
        )""",
    ]),
    (6, [
        # 显示/导出用：保留上传时的原始文件名（问题2-D）
        """ALTER TABLE images ADD COLUMN filename TEXT""",
    ]),
]


def get_db_path(db_path=None):
    if db_path is not None:
        return str(db_path)
    data_dir = Config.DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    return str(Path(data_dir) / 'fluro.db')


def get_conn(db_path=None):
    conn = sqlite3.connect(get_db_path(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_db(db_path=None):
    """建表并执行未应用的迁移。幂等。"""
    conn = get_conn(db_path)
    try:
        conn.execute('CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)')
        row = conn.execute('SELECT version FROM schema_version ORDER BY version DESC LIMIT 1').fetchone()
        current = row['version'] if row else 0
        for version, statements in sorted(MIGRATIONS):
            if version > current:
                for stmt in statements:
                    conn.execute(stmt)
                conn.execute('INSERT INTO schema_version (version) VALUES (?)', (version,))
        conn.commit()
    finally:
        conn.close()


def execute(db_path, sql, params=()):
    conn = get_conn(db_path)
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def execute_many(db_path, sql, seq_params):
    conn = get_conn(db_path)
    try:
        conn.executemany(sql, seq_params)
        conn.commit()
    finally:
        conn.close()


def query(db_path, sql, params=()):
    conn = get_conn(db_path)
    try:
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def query_one(db_path, sql, params=()):
    rows = query(db_path, sql, params)
    return rows[0] if rows else None


def get_setting(db_path, key, default=None):
    row = query_one(db_path, 'SELECT value FROM settings WHERE key=?', (key,))
    return row['value'] if row else default


def set_setting(db_path, key, value):
    execute(db_path,
            'INSERT INTO settings (key, value) VALUES (?,?) '
            'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
            (key, str(value) if value is not None else None))
