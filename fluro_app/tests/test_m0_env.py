# -*- coding: utf-8 -*-
"""M0 环境与骨架测试：依赖可导入、启动脚本正确、应用可创建并响应首页。"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_dependencies_installed():
    for mod in ['flask', 'numpy', 'cv2', 'scipy']:
        importlib.import_module(mod)


def test_start_bat_exists_and_launches_server():
    bat = os.path.join(ROOT, 'start.bat')
    assert os.path.exists(bat), 'start.bat 不存在'
    with open(bat, encoding='utf-8') as f:
        content = f.read()
    assert 'server.py' in content, 'start.bat 未引用 server.py'
    assert 'python' in content.lower(), 'start.bat 未调用 python'


def test_app_creates_and_index_serves():
    from app import create_app
    app = create_app({'TESTING': True})
    client = app.test_client()
    r = client.get('/')
    assert r.status_code == 200, f'首页返回 {r.status_code}'
    assert '荧光浓度建模'.encode('utf-8') in r.data


def test_static_files_served():
    from app import create_app
    app = create_app({'TESTING': True})
    client = app.test_client()
    r = client.get('/static/index.html')
    assert r.status_code == 200


def test_server_module_importable():
    import server  # noqa: F401


def test_config_has_defaults():
    from app.config import Config
    assert Config.DEFAULT_FILTER in ('gaussian', 'median')
    assert Config.CONFIDENCE == 0.95
    assert Config.PORT >= 8000


def test_server_real_startup():
    """真实拉起 server.py 子进程，访问首页，验证双击启动路径可用。"""
    import subprocess
    import time
    import urllib.request

    port = 8791
    env = dict(os.environ)
    env['FLURO_PORT'] = str(port)
    env['FLURO_NO_BROWSER'] = '1'
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, 'server.py')],
        cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        ok = False
        for _ in range(40):
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=1) as r:
                    body = r.read()
                    assert r.status == 200
                    assert '荧光浓度建模'.encode('utf-8') in body
                    ok = True
                    break
            except Exception:
                time.sleep(0.25)
        assert ok, 'server.py 子进程未能启动并响应首页'
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
