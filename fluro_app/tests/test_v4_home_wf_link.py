# -*- coding: utf-8 -*-
"""第四批问题1：首页「最近检测工作流（每步处理图像）」与检测工作流界面联动、实时刷新。

前端契约测试（无需浏览器）：
- 工作流卡片缩略图为可点击按钮（data-step / data-image），点击后调用
  switchView('workflow') + FluroWorkflow.openStep(imageId, step) 跳转对应步骤；
- workflow.js 导出 openStep（选中图 + 切换步骤面板 + 预处理步骤自动重跑）；
- 流水线完成后（runPipeline 成功）会调用 FluroHome.refresh() 实时刷新首页；
- 首页 refresh 仅当 home 视图可见时执行，且带 5s 轮询；
- home.js / workflow.js 通过 node --check 语法校验。
"""
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / 'static'
INDEX = STATIC / 'index.html'
HOME_JS = STATIC / 'js' / 'views' / 'home.js'
WF_JS = STATIC / 'js' / 'views' / 'workflow.js'


def test_home_wf_card_exists_in_index():
    """首页存在工作流卡片容器，工作流存在 6 步步骤条。"""
    html = INDEX.read_text(encoding='utf-8')
    assert 'id="home-wf-card"' in html
    assert 'id="home-wf"' in html
    assert 'id="home-wf-meta"' in html
    assert html.count('class="wf-step') >= 6


def test_home_js_click_link_to_workflow():
    """首页缩略图点击 → 跳转检测工作流并选中图（openStep 联动）。"""
    js = HOME_JS.read_text(encoding='utf-8')
    # 缩略图为可点击按钮，带步骤与图片 id
    assert "data-step=" in js
    assert "data-image=" in js
    # 点击处理：切到工作流视图 + 调 openStep
    assert "switchView('workflow')" in js
    assert "openStep(imageId, step)" in js
    # 实时刷新：仅首页可见时刷新 + 5s 轮询
    assert 'view-home' in js
    assert 'setInterval(refresh, 5000)' in js


def test_workflow_js_exports_open_step_and_home_refresh_hook():
    """workflow.js 导出 openStep；流水线完成后触发首页实时刷新。"""
    js = WF_JS.read_text(encoding='utf-8')
    assert 'openStep: openStep' in js
    # openStep：选中缩略图 + 切面板 + 预处理步骤自动重跑
    assert 'function openStep(id, step)' in js
    assert 'markThumbSelected(id)' in js
    assert 'showWfPanel(step)' in js
    assert 'runPreprocess(true)' in js
    # runPipeline 完成后刷新首页（实时显示最新每步图）
    assert 'FluroHome.refresh' in js


@pytest.mark.skipif(shutil.which('node') is None, reason='node 未安装')
@pytest.mark.parametrize('js', [HOME_JS, WF_JS])
def test_js_syntax(js):
    """home.js / workflow.js 语法合法（node --check）。"""
    r = subprocess.run(['node', '--check', str(js)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f'{js.name} 语法错误：\n{r.stderr}'
