# -*- coding: utf-8 -*-
"""服务入口：启动 Flask 本地服务并自动打开浏览器。"""
import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app  # noqa: E402


def main():
    cfg = {}
    if os.environ.get('FLURO_DATA_DIR'):
        cfg['DATA_DIR'] = os.environ['FLURO_DATA_DIR']
    app = create_app(cfg if cfg else None)
    port = app.config['PORT']
    url = f'http://127.0.0.1:{port}'
    print(f'荧光浓度建模 APP 已启动：{url}')
    print(f'数据目录：{app.config["DATA_DIR"]}')
    threading.Timer(1.2, lambda: webbrowser.open(url)).start() if os.environ.get('FLURO_NO_BROWSER') != '1' else None
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
