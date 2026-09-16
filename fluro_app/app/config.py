# -*- coding: utf-8 -*-
"""全局配置：数据目录、端口、默认参数。"""
import os
import socket
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # fluro_app/
DEFAULT_DATA_DIR = BASE_DIR / 'data'


def find_free_port(preferred=8000):
    """在 preferred 起 20 个端口内找空闲端口（避免重复启动/端口占用）。"""
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    return preferred


class Config:
    # 环境变量 FLURO_PORT 用于测试等场景固定端口
    PORT = int(os.environ.get('FLURO_PORT', find_free_port(8000)))
    DATA_DIR = DEFAULT_DATA_DIR
    STATIC_DIR = BASE_DIR / 'static'

    # 图像与模型默认参数（后续阶段使用）
    DEFAULT_FILTER = 'gaussian'   # gaussian | median
    DEFAULT_KERNEL = 5            # 滤波核大小
    CONFIDENCE = 0.95             # 不确定度置信水平
    LIMIT_LOWER = 0.0             # 判定下限
    LIMIT_UPPER = None            # 判定上限（None 表示未设置）
    UNIT = 'ng/mL'                # 浓度单位
