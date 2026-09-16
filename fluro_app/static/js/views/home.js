/* views/home.js：首页总览（M7） */
(function (global) {
  'use strict';

  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  var judgeMap = {
    within: ['正常范围', 'st-ok'],
    above: ['高于上限', 'st-error'],
    below: ['低于下限', 'st-error'],
    borderline: ['临界', 'st-attention'],
  };

  function renderLast(d) {
    const el = document.getElementById('home-last');
    if (!d) {
      el.innerHTML = '<div class="empty">暂无检测记录。<br>上传检测图并完成流水线后，结果会出现在这里。</div>';
      return;
    }
    const jm = judgeMap[d.status] || [d.status, 'st-uploaded'];
    el.innerHTML =
      '<div class="det-conc" style="font-size:26px;">' + Number(d.conc).toFixed(2) + ' <span class="det-unit">ng/mL</span></div>'
      + '<div class="det-u">U(95%) = ±' + Number(d.u).toFixed(2)
      + '　区间 [' + (d.conc - d.u).toFixed(2) + ', ' + (d.conc + d.u).toFixed(2) + ']</div>'
      + '<div class="det-judge" style="margin-top:6px;"><span class="gstatus ' + jm[1] + '">' + jm[0] + '</span></div>'
      + '<div class="det-meta">' + esc(d.created_at || '') + '　模型：' + esc(d.model_name || '-') + '</div>';
  }

  function renderCalib(s) {
    const el = document.getElementById('home-calib');
    if (!s.calibrated) {
      el.innerHTML = '<div class="empty">尚未建立标定曲线。<br>到「标定建模」页上传标定图并完成分组拟合。</div>';
      return;
    }
    el.innerHTML =
      '<div style="font-size:22px;font-weight:600;color:#0f7d72;">' + s.n_groups + ' 组浓度</div>'
      + '<div class="det-meta">有效数据点 ' + s.n_points + ' 个</div>'
      + (s.model_lod != null ? '<div class="det-meta">LOD = ' + s.model_lod + ' ng/mL</div>' : '');
  }

  function renderModel(s) {
    const el = document.getElementById('home-model');
    if (!s.model_name) {
      el.innerHTML = '<div class="empty">尚无生效模型。</div>';
      return;
    }
    el.innerHTML =
      '<div style="font-size:18px;font-weight:600;">' + esc(s.model_name) + '</div>'
      + '<div class="det-meta">类型：' + esc(s.model_type || '-') + '</div>'
      + '<div class="det-meta">R² = ' + (s.model_r2 == null ? '-' : Number(s.model_r2).toFixed(4)) + '</div>';
  }

  function renderToday(s) {
    const el = document.getElementById('home-today');
    el.innerHTML =
      '<div style="font-size:22px;font-weight:600;color:#1d6fb8;">' + s.today_count + ' 张</div>'
      + '<div class="det-meta">今日上传图片</div>'
      + '<div style="font-size:16px;font-weight:600;margin-top:10px;">' + s.total_detections + ' 条</div>'
      + '<div class="det-meta">累计检测记录</div>'
      + '<div class="det-meta" style="margin-top:8px;">当前批次：' + esc(s.current_batch || '-') + '</div>';
  }

  function renderWf(s) {
    const el = document.getElementById('home-wf');
    const meta = document.getElementById('home-wf-meta');
    const wf = s.workflow_steps;
    if (!wf) {
      el.innerHTML = '<div class="empty">暂无检测记录。完成一次检测后，这里会展示该图在「上传→预处理→通道分离→ROI」各步的处理图像。</div>';
      meta.textContent = '';
      return;
    }
    meta.textContent = '图片 #' + wf.image_id;
    let html = '<div style="display:flex;gap:10px;flex-wrap:wrap;">';
    const items = [
      { label: '① 上传原图', url: wf.upload },
    ];
    if (wf.preprocess) items.push({ label: '② 预处理后', url: wf.preprocess });
    if (wf.channels) {
      (['r', 'g', 'b']).forEach(function (ch) {
        if (wf.channels[ch]) items.push({ label: '③ 通道 ' + ch.toUpperCase(), url: wf.channels[ch] });
      });
    }
    if (wf.roi) items.push({ label: '④ ROI 叠加', url: wf.roi });
    items.forEach(function (it) {
      html += '<div style="text-align:center;max-width:150px;">'
        + '<a href="' + it.url + '" target="_blank" rel="noopener">'
        + '<img src="' + it.url + '?t=' + Date.now() + '" alt="' + esc(it.label) + '"'
        + ' style="width:150px;height:110px;object-fit:cover;border:1px solid #d9e2ea;border-radius:6px;display:block;"></a>'
        + '<div style="font-size:12px;color:#55636f;margin-top:4px;">' + esc(it.label) + '</div>'
        + '</div>';
    });
    html += '</div>';
    el.innerHTML = html;
  }

  async function refresh() {
    try {
      const s = await global.API.get('/api/home/summary');
      renderLast(s.last_detection);
      renderCalib(s);
      renderModel(s);
      renderToday(s);
      renderWf(s);
    } catch (e) {
      document.getElementById('home-last').innerHTML = '<div class="empty">加载失败：' + esc(e.message) + '</div>';
    }
  }

  function init() {
    refresh();
  }

  function onView() { refresh(); }

  global.FluroHome = { init: init, onView: onView, refresh: refresh };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
