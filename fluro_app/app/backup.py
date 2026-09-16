# -*- coding: utf-8 -*-
"""本地数据备份与恢复（M8）：把 data 目录整体复制到 backups/<时间戳>，支持列出与恢复。"""
import shutil
import time
from pathlib import Path

_SUBDIRS = ('images', 'thumbnails', 'processed', 'roi_overlays')


def _default_backups_dir(data_dir):
    return Path(data_dir).parent / 'backups'


def _subdirs(data_dir):
    d = Path(data_dir)
    return [d / s for s in _SUBDIRS] + [d / 'fluro.db']


def create_backup(data_dir, backups_dir=None, label=None):
    """创建备份。返回 {name, created_at, size, ok}。"""
    src = Path(data_dir)
    if not (src / 'fluro.db').exists():
        raise FileNotFoundError('数据目录中不存在 fluro.db，无法备份')
    backups_dir = Path(backups_dir) if backups_dir else _default_backups_dir(src)
    backups_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    base = f'backup_{stamp}' + (f'_{label}' if label else '')
    name = base
    dst = backups_dir / name
    i = 2
    while dst.exists():   # 同一秒多次备份时加序号
        name = f'{base}_{i}'
        dst = backups_dir / name
        i += 1
    dst.mkdir(parents=True, exist_ok=True)
    # 复制数据库与子目录（不存在的跳过）
    shutil.copy2(src / 'fluro.db', dst / 'fluro.db')
    for sub in _SUBDIRS:
        s = src / sub
        if s.exists():
            shutil.copytree(s, dst / sub)
    size = sum(f.stat().st_size for f in dst.rglob('*') if f.is_file())
    return {'name': name, 'created_at': stamp, 'size': size, 'ok': True}


def list_backups(data_dir, backups_dir=None):
    backups_dir = Path(backups_dir) if backups_dir else _default_backups_dir(data_dir)
    if not backups_dir.exists():
        return []
    out = []
    for d in sorted(backups_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / 'fluro.db').exists():
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            out.append({'name': d.name, 'created_at': d.name.replace('backup_', ''), 'size': size})
    return out


def restore_backup(data_dir, name, backups_dir=None):
    """把指定备份恢复回 data 目录。先自动创建当前数据的安全备份。"""
    src = Path(data_dir)
    backups_dir = Path(backups_dir) if backups_dir else _default_backups_dir(src)
    backup = backups_dir / name
    if not (backup / 'fluro.db').exists():
        raise FileNotFoundError(f'备份不存在：{name}')
    # 恢复前安全备份当前数据
    safe = create_backup(data_dir, backups_dir=backups_dir, label='before_restore')
    # 清空当前数据目录（保留目录本身）
    for item in src.iterdir():
        if item.name in ('images', 'thumbnails', 'processed', 'roi_overlays') or item.name == 'fluro.db':
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink(missing_ok=True)
    shutil.copy2(backup / 'fluro.db', src / 'fluro.db')
    for sub in _SUBDIRS:
        s = backup / sub
        if s.exists():
            shutil.copytree(s, src / sub)
    return {'ok': True, 'restored': name, 'safe_backup': safe['name']}
