/* views/modeling.js：标定建模视图（M5）
   浓度分组管理、4 模型拟合对比、曲线 SVG、模型库 */
(function (global) {
  'use strict';

  var curResults = null;      // fit 结果
  var curFeature = 'hue';
  var selectedType = null;    // 当前选中的模型类型
  var dataSnapshot = null;    // calibration/data 快照（保存模型时附 source_snapshot）

  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  async function loadData() {
    const res = await global.API.get('/api/calibration/data');
    dataSnapshot = res;
    renderPending(res.groups);
    renderGroups(res.groups);
    renderSaved(res.models);
    renderCurve(res.active_model);
  }

  function renderPending(groups) {
    const el = document.getElementById('md-pending');
    const cal = (dataSnapshot && dataSnapshot.pending_cal) || [];
    if (!cal.length) {
      el.innerHTML = '<div class="hint">无待分组的标定图（上传标定图并完成流水线后，可在此加入浓度分组）。</div>';
      return;
    }
    let html = '<div class="pending-title">待分组的标定图：</div><table class="md-table"><thead><tr><th>图</th><th>已知浓度</th><th>色相</th><th>加入分组</th></tr></thead><tbody>';
    cal.forEach(function (c) {
      const opts = groups.map(function (g) { return '<option value="' + g.id + '">' + g.conc + '（' + g.name + '）</option>'; }).join('');
      html += '<tr><td>#' + c.id + '</td><td>' + c.known_conc + '</td><td>' + c.hue + '</td>'
        + '<td><select class="grp-sel" data-img="' + c.id + '">'
        + '<option value="">选择分组...</option>' + opts
        + '<option value="__new__">新建分组</option></select></td></tr>';
    });
    html += '</tbody></table>';
    el.innerHTML = html;
    el.querySelectorAll('.grp-sel').forEach(function (sel) {
      sel.addEventListener('change', async function () {
        const imgId = parseInt(sel.dataset.img, 10);
        let gid = sel.value;
        if (gid === '__new__') {
          const conc = prompt('新建分组浓度（ng/mL）：', '');
          if (!conc || isNaN(parseFloat(conc))) { sel.value = ''; return; }
          const gres = await global.API.post('/api/calibration/groups', { conc: parseFloat(conc) });
          gid = gres.id;
        }
        if (!gid) { sel.value = ''; return; }
        try {
          await global.API.post('/api/calibration/points', { image_id: imgId, group_id: parseInt(gid, 10) });
          await loadData();
        } catch (e) {
          alert('加入分组失败：' + e.message);
          sel.value = '';
        }
      });
    });
  }

  function renderGroups(groups) {
    const el = document.getElementById('md-data');
    if (!groups || groups.length === 0) {
      el.innerHTML = '<div class="empty">暂无标定数据。请上传「标定」图并完成流水线（ROI/特征），再创建浓度分组加入。</div>';
      return;
    }
    let html = '<table class="md-table"><thead><tr><th>浓度</th><th>分组</th><th>点数</th><th>均值(当前特征)</th><th>SD</th><th>操作</th></tr></thead><tbody>';
    groups.forEach(function (g) {
      const m = g.mean ? g.mean[curFeature] : null;
      const s = g.sd ? g.sd[curFeature] : null;
      html += '<tr><td>' + esc(g.conc) + '</td><td>' + esc(g.name) + '</td><td>' + g.n + '</td>'
        + '<td>' + (m == null ? '-' : m) + '</td><td>' + (s == null ? '-' : s) + '</td>'
        + '<td><button class="btn small" data-del="' + g.id + '">删除分组</button></td></tr>';
    });
    html += '</tbody></table>';
    el.innerHTML = html;
    el.querySelectorAll('[data-del]').forEach(function (b) {
      b.addEventListener('click', async function () {
        if (!confirm('删除该分组及其数据点？')) return;
        await global.API.post('/api/calibration/groups/' + b.dataset.del + '/delete', {});
        await global.API.get('/api/calibration/data');
        loadData();
      });
    });
  }

  function renderSaved(models) {
    const el = document.getElementById('md-saved');
    if (!models || models.length === 0) {
      el.innerHTML = '<div class="empty">尚无保存的模型。</div>';
      return;
    }
    let html = '<table class="md-table"><thead><tr><th>名称</th><th>类型</th><th>R²</th><th>RMSE</th><th>LOD</th><th>状态</th><th>操作</th></tr></thead><tbody>';
    models.forEach(function (m) {
      const met = safeJson(m.metrics_json);
      html += '<tr><td>' + esc(m.name) + '</td><td>' + esc(m.type) + '</td>'
        + '<td>' + (met.r2 == null ? '-' : met.r2.toFixed(4)) + '</td>'
        + '<td>' + (met.rmse == null ? '-' : met.rmse.toFixed(4)) + '</td>'
        + '<td>' + (met.lod == null ? '-' : met.lod) + '</td>'
        + '<td>' + (m.is_active ? '<span class="pb pb-ok">生效中</span>' : '') + '</td>'
        + '<td><button class="btn small" data-act="' + m.id + '">设为生效</button> '
        + '<button class="btn small" data-del="' + m.id + '">删除</button></td></tr>';
    });
    html += '</tbody></table>';
    el.innerHTML = html;
    el.querySelectorAll('[data-act]').forEach(function (b) {
      b.addEventListener('click', async function () {
        await global.API.post('/api/models/' + b.dataset.act + '/activate', {});
        loadData();
      });
    });
    el.querySelectorAll('[data-del]').forEach(function (b) {
      b.addEventListener('click', async function () {
        if (!confirm('删除该模型？')) return;
        await global.API.post('/api/models/' + b.dataset.del + '/delete', {});
        loadData();
      });
    });
  }

  function safeJson(s) {
    try { return JSON.parse(s || '{}'); } catch (e) { return {}; }
  }

  async function runFit() {
    curFeature = document.getElementById('md-feature').value;
    const status = document.getElementById('md-status');
    status.textContent = '拟合中...';
    try {
      const res = await global.API.post('/api/calibration/fit', { feature: curFeature });
      curResults = res;
      selectedType = res.best;
      renderModels(res);
      renderCurveFromResults(res);
      status.textContent = '拟合完成：' + res.n + ' 个数据点，最佳 ' + (res.best || '-');
    } catch (e) {
      status.textContent = '拟合失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  function renderModels(res) {
    const el = document.getElementById('md-models');
    const order = ['linear', 'poly2', 'exp', '4pl'];
    const names = { linear: '线性', poly2: '二次多项式', exp: '指数', '4pl': '四参数逻辑 4PL' };
    let html = '<table class="md-table"><thead><tr><th></th><th>模型</th><th>R²</th><th>RMSE</th><th>LOD</th><th>状态</th></tr></thead><tbody>';
    order.forEach(function (mt) {
      const r = res.results[mt];
      if (!r) return;
      const isBest = res.best === mt;
      html += '<tr class="' + (selectedType === mt ? 'sel' : '') + '" data-type="' + mt + '" style="cursor:pointer;">'
        + '<td>' + (isBest ? '<span class="pb pb-ok">推荐</span>' : '') + '</td>'
        + '<td>' + names[mt] + '</td>';
      if (r.error) {
        html += '<td colspan="3" class="err-text">' + esc(r.error) + '</td>';
      } else {
        html += '<td>' + r.r2.toFixed(4) + '</td><td>' + r.rmse.toFixed(4) + '</td><td>' + (r.lod == null ? '-' : r.lod) + '</td>';
      }
      html += '<td>' + (selectedType === mt ? '选中' : '') + '</td></tr>';
    });
    html += '</tbody></table>';
    html += '<div class="hint">点击行选中模型，再点「保存所选为生效模型」。</div>';
    el.innerHTML = html;
    el.querySelectorAll('tr[data-type]').forEach(function (tr) {
      tr.addEventListener('click', function () {
        selectedType = tr.dataset.type;
        renderModels(res);
        renderCurveFromResults(res);
      });
    });
  }

  function renderCurve(activeModel) {
    if (!activeModel) {
      document.getElementById('md-curve').innerHTML = '<div class="empty">尚无生效模型，拟合后保存一个。</div>';
      document.getElementById('md-curve-title').textContent = '';
      return;
    }
    const params = safeJson(activeModel.params_json);
    const met = safeJson(activeModel.metrics_json);
    const src = safeJson(activeModel.source_snapshot_json);
    const data = src.data || [];
    const curve = buildCurve(activeModel.type, params, data);
    drawSvg('md-curve', data, curve, activeModel.type, met);
    document.getElementById('md-curve-title').textContent = '生效模型：' + activeModel.name + '（R²=' + (met.r2 == null ? '-' : met.r2.toFixed(4)) + '）';
  }

  function renderCurveFromResults(res) {
    const r = res.results[selectedType];
    if (!r || r.error) {
      document.getElementById('md-curve').innerHTML = '<div class="empty">该模型不可用。</div>';
      return;
    }
    document.getElementById('md-curve-title').textContent = '候选：' + selectedType + '（R²=' + r.r2.toFixed(4) + '，LOD=' + (r.lod == null ? '-' : r.lod) + '）';
    drawSvg('md-curve', res.data, r.curve, selectedType, r);
  }

  function buildCurve(type, params, data) {
    // 后端已提供 curve 点；这里用保存的 params 重新生成（与后端一致的最小实现）
    const xs = data.map(function (d) { return d[0]; });
    const xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
    const n = 80;
    const pts = [];
    for (let i = 0; i < n; i++) {
      const x = xmin + (xmax - xmin) * i / (n - 1);
      let y = 0;
      if (type === 'linear') y = params.a * x + params.b;
      else if (type === 'poly2') y = params.a * x * x + params.b * x + params.c;
      else if (type === 'exp') y = params.a * Math.exp(params.b * x) + params.c;
      else if (type === '4pl') y = params.d + (params.a - params.d) / (1 + Math.pow(x / params.c, params.b));
      pts.push([x, y]);
    }
    return pts;
  }

  function drawSvg(elId, data, curve, type, met) {
    const el = document.getElementById(elId);
    const W = 760, H = 380, padL = 70, padR = 20, padT = 20, padB = 46;
    if (!data || data.length === 0) { el.innerHTML = '<div class="empty">无数据点。</div>'; return; }
    const all = data.concat(curve || []);
    const xmin = Math.min.apply(null, all.map(function (p) { return p[0]; }));
    const xmax = Math.max.apply(null, all.map(function (p) { return p[0]; }));
    const ymin = Math.min.apply(null, all.map(function (p) { return p[1]; }));
    const ymax = Math.max.apply(null, all.map(function (p) { return p[1]; }));
    const xr = xmax - xmin || 1, yr = ymax - ymin || 1;
    const X = function (x) { return padL + (x - xmin) / xr * (W - padL - padR); };
    const Y = function (y) { return H - padB - (y - ymin) / yr * (H - padT - padB); };

    let svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;max-width:820px;background:#fff;">';
    // 网格
    for (let i = 0; i <= 5; i++) {
      const gx = padL + i / 5 * (W - padL - padR);
      svg += '<line x1="' + gx + '" y1="' + padT + '" x2="' + gx + '" y2="' + (H - padB) + '" stroke="#eef2f5"/>';
      const val = xmin + i / 5 * xr;
      svg += '<text x="' + gx + '" y="' + (H - padB + 18) + '" font-size="10" fill="#8a97a3" text-anchor="middle">' + val.toFixed(val < 100 ? 2 : 1) + '</text>';
    }
    for (let j = 0; j <= 5; j++) {
      const gy = H - padB - j / 5 * (H - padT - padB);
      svg += '<line x1="' + padL + '" y1="' + gy + '" x2="' + (W - padR) + '" y2="' + gy + '" stroke="#eef2f5"/>';
      const val = ymin + j / 5 * yr;
      svg += '<text x="' + (padL - 8) + '" y="' + (gy + 3) + '" font-size="10" fill="#8a97a3" text-anchor="end">' + val.toFixed(val < 100 ? 2 : 1) + '</text>';
    }
    // 曲线
    if (curve && curve.length) {
      let d = '';
      curve.forEach(function (p, i) {
        d += (i === 0 ? 'M' : 'L') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1);
      });
      svg += '<path d="' + d + '" fill="none" stroke="#0f9d8f" stroke-width="2"/>';
    }
    // 散点
    data.forEach(function (p) {
      svg += '<circle cx="' + X(p[0]).toFixed(1) + '" cy="' + Y(p[1]).toFixed(1) + '" r="4" fill="#1d6fb8"/>';
    });
    svg += '<text x="' + (W / 2) + '" y="' + (H - 8) + '" font-size="11" fill="#55636f" text-anchor="middle">浓度（ng/mL）</text>';
    svg += '<text x="16" y="' + (H / 2) + '" font-size="11" fill="#55636f" text-anchor="middle" transform="rotate(-90 16 ' + (H / 2) + ')">特征值</text>';
    svg += '</svg>';
    el.innerHTML = svg;
  }

  async function saveModel() {
    if (!curResults || !selectedType) {
      document.getElementById('md-status').textContent = '请先拟合并选中一个模型';
      return;
    }
    const r = curResults.results[selectedType];
    if (!r || r.error) {
      document.getElementById('md-status').textContent = '该模型不可用';
      return;
    }
    const name = prompt('模型名称：', selectedType + '-' + new Date().toLocaleDateString());
    if (!name) return;
    const snapshot = {
      feature: curFeature,
      data: curResults.data,
      n: curResults.n,
      saved_at: new Date().toISOString(),
    };
    try {
      const res = await global.API.post('/api/models', {
        name: name, type: selectedType, params: r.params,
        metrics: { r2: r.r2, rmse: r.rmse, lod: r.lod },
        source_snapshot: snapshot,
      });
      document.getElementById('md-status').textContent = '已保存并设为生效模型：' + name;
      await loadData();
      renderCurveFromResults(curResults);
    } catch (e) {
      document.getElementById('md-status').textContent = '保存失败：' + e.message;
    }
  }

  var initialized = false;

  function init() {
    if (initialized) return;
    initialized = true;
    document.getElementById('md-fit').addEventListener('click', runFit);
    document.getElementById('md-save').addEventListener('click', saveModel);
    document.getElementById('md-feature').addEventListener('change', function () {
      curFeature = document.getElementById('md-feature').value;
      loadData();
    });
    loadData();
  }

  function onView() {
    if (!initialized) init();
    else loadData();
  }

  global.FluroModeling = { init: init, onView: onView };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
