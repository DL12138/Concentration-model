# -*- coding: utf-8 -*-
"""Flask 应用工厂：注册各模块蓝图。"""
import os
from pathlib import Path
from flask import Flask, send_from_directory
from .config import Config


def create_app(config=None):
    app = Flask(__name__, static_folder=str(Config.STATIC_DIR), static_url_path='/static')
    app.config.from_object(Config)
    if config:
        app.config.update(config)

    # 初始化数据目录与数据库
    os.makedirs(app.config['DATA_DIR'], exist_ok=True)
    from . import database as db
    db.init_db(Path(app.config['DATA_DIR']) / 'fluro.db')

    @app.route('/')
    def index():
        return send_from_directory(app.config['STATIC_DIR'], 'index.html')

    # 蓝图注册（按模块逐步加入）
    from .routes import api_workflow, api_model, api_settings
    app.register_blueprint(api_workflow.bp)
    app.register_blueprint(api_model.bp)
    app.register_blueprint(api_settings.bp)

    return app
