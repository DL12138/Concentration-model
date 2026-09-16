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
    let html = '<table class="md-table"><thead><tr><th>浓度</th><th>分组</th><th>重复</th><th>各重复值（' + esc(curFeature) + '）</th><th>均值</th><th>SD</th><th>CV%</th><th>操作</th></tr></thead><tbody>';
    groups.forEach(function (g) {
      const m = g.mean ? g.mean[curFeature] : null;
      const s = g.sd ? g.sd[curFeature] : null;
      const cv = g.cv ? g.cv[curFeature] : null;
      // 每张重复图的结果（按 replicate 排序）
      const reps = (g.points || []).filter(function (p) { return p.included; })
        .map(function (p) {
          const v = (p.features && p.features[curFeature]) != null ? p.features[curFeature] : p.feature_value;
          const repNo = p.replicate || '-';
          return (v == null ? '-' : Number(v).toFixed(4)) + '<span style="color:#8a97a3;">(#' + repNo + ')</span>';
        })
        .join('　');
      html += '<tr><td>' + esc(g.conc) + (g.unit && g.unit !== 'ng/mL' ? ' ' + esc(g.unit) : '') + '</td>'
        + '<td>' + esc(g.name) + '</td><td>' + g.n + '</td>'
        + '<td>' + (reps || '-') + '</td>'
        + '<td>' + (m == null ? '-' : Number(m).toFixed(4)) + '</td>'
        + '<td>' + (s == null ? '-' : Number(s).toFixed(4)) + '</td>'
        + '<td>' + (cv == null ? '-' : cv) + '</td>'
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
      // 问题2-H：先持久化预处理设置，拟合/检测共用同一口径
      await global.API.post('/api/modeling/preprocess', {
        iqr: document.getElementById('md-iqr').checked,
        log_conc: document.getElementById('md-log').checked,
        zscore: document.getElementById('md-zscore').checked,
      });
      const res = await global.API.post('/api/calibration/fit', { feature: curFeature });
      curResults = res;
      selectedType = res.best;
      renderModels(res);
      renderCurveFromResults(res);
      const prepInfo = res.preprocess && res.preprocess.y_std ? '（Z-score 已标准化）' : '';
      status.textContent = '拟合完成：' + res.n + ' 个数据点，最佳 ' + (res.best || '-')
        + (res.preprocess && res.preprocess.removed ? '，IQR 剔除 ' + res.preprocess.removed + ' 个' : '') + prepInfo;
    } catch (e) {
      status.textContent = '拟合失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  function renderModels(res) {
    const el = document.getElementById('md-models');
    const order = ['linear', 'poly2', 'log', 'exp', '4pl', 'pls', 'svr', 'rf'];
    const names = {
      linear: '线性', poly2: '二次多项式', log: '对数', exp: '指数',
      '4pl': '四参数逻辑 4PL', pls: 'PLSR', svr: 'SVR（对比）', rf: '随机森林（对比）',
    };
    let html = '<table class="md-table"><thead><tr><th></th><th>模型</th><th>R²</th><th>RMSE</th><th>LOD</th><th>状态</th></tr></thead><tbody>';
    order.forEach(function (mt) {
      const r = res.results[mt];
      if (!r) return;
      const isBest = res.best === mt;
      html += '<tr class="' + (selectedType === mt ? 'sel' : '') + '" data-type="' + mt + '" style="cursor:pointer;">'
        + '<td>' + (isBest ? '<span class="pb pb-ok">推荐</span>' : '') + '</td>'
        + '<td>' + names[mt] + (r.compare ? ' <span class="gstatus st-attention">仅对比</span>' : '') + '</td>';
      if (r.error) {
        html += '<td colspan="3" class="err-text">' + esc(r.error) + '</td>';
      } else {
        html += '<td>' + r.r2.toFixed(4) + '</td><td>' + r.rmse.toFixed(4) + '</td><td>' + (r.lod == null ? '-' : r.lod) + '</td>';
      }
      html += '<td>' + (selectedType === mt ? '选中' : '') + '</td></tr>';
    });
    html += '</tbody></table>';
    html += '<div class="hint">点击行选中模型，再点「保存所选为生效模型」；SVR/随机森林仅作对比，不可保存为生效模型。</div>';
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
    svg += '<text x="' + (W / 2) + '" y="' + (H - 8) + '" font-size="11" fill="#55636f" text-anchor="middle">浓度（' + (curResults && curResults.unit || 'ng/mL') + '）</text>';
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
    if (r.compare) {
      document.getElementById('md-status').textContent = 'SVR/随机森林为对比模型，暂不支持保存为生效模型（无可解析反解）';
      return;
    }
    const name = prompt('模型名称：', selectedType + '-' + new Date().toLocaleDateString());
    if (!name) return;
    const snapshot = {
      feature: curFeature,
      data: curResults.data,
      n: curResults.n,
      unit: curResults.unit || 'ng/mL',
      preprocess: curResults.preprocess || {},   // 问题2-H：预处理元数据（检测反解用）
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

  async function savePreprocess() {
    // 问题2-H：拟合前先把预处理设置持久化，拟合/检测共用同一套口径
    return global.API.post('/api/modeling/preprocess', {
      iqr: document.getElementById('md-iqr').checked,
      log_conc: document.getElementById('md-log').checked,
      zscore: document.getElementById('md-zscore').checked,
    });
  }

  function renderExplore(d) {
    // 问题2-I：SVG 散点 + 每浓度组箱线（min/q1/med/q3/max）
    const el = document.getElementById('md-explore-plot');
    const W = 560, H = 240, padL = 46, padR = 12, padT = 14, padB = 30;
    const pts = d.points || [];
    const xs = pts.map(function (p) { return p[0]; });
    const ys = pts.map(function (p) { return p[1]; });
    const xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
    const ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    const X = function (v) { return padL + (v - xmin) / (xmax - xmin || 1) * (W - padL - padR); };
    const Y = function (v) { return padT + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - padT - padB); };
    let svg = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '">';
    svg += '<line x1="' + padL + '" y1="' + (H - padB) + '" x2="' + (W - padR) + '" y2="' + (H - padB) + '" stroke="#cfd8df"/>';
    svg += '<line x1="' + padL + '" y1="' + padT + '" x2="' + padL + '" y2="' + (H - padB) + '" stroke="#cfd8df"/>';
    svg += '<text x="' + (W / 2) + '" y="' + (H - 8) + '" font-size="11" fill="#55636f" text-anchor="middle">浓度</text>';
    svg += '<text x="14" y="' + (H / 2) + '" font-size="11" fill="#55636f" text-anchor="middle" transform="rotate(-90 14 ' + (H / 2) + ')">特征值</text>';
    // 箱线：每组 conc → 竖线 min/max + 箱 q1..q3 + 中位横线
    (d.box || []).forEach(function (g) {
      const vals = g.values.slice().sort(function (a, b) { return a - b; });
      const q = function (pct) {
        const pos = (vals.length - 1) * pct;
        const lo = Math.floor(pos), hi = Math.ceil(pos);
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo);
      };
      const cx = X(g.conc);
      const vmin = vals[0], vmax = vals[vals.length - 1], q1 = q(0.25), q3 = q(0.75), med = q(0.5);
      svg += '<line x1="' + cx + '" y1="' + Y(vmin) + '" x2="' + cx + '" y2="' + Y(vmax) + '" stroke="#94a5b4" stroke-width="1.5"/>';
      svg += '<rect x="' + (cx - 6) + '" y="' + Y(q3) + '" width="12" height="' + (Y(q1) - Y(q3)) + '" fill="#cfe0f2" stroke="#4a7fd4"/>';
      svg += '<line x1="' + (cx - 8) + '" y1="' + Y(med) + '" x2="' + (cx + 8) + '" y2="' + Y(med) + '" stroke="#1d6fb8" stroke-width="2"/>';
    });
    // 散点
    pts.forEach(function (p) {
      svg += '<circle cx="' + X(p[0]).toFixed(1) + '" cy="' + Y(p[1]).toFixed(1) + '" r="3.5" fill="#e05656"/>';
    });
    svg += '</svg>';
    el.innerHTML = svg;
    el.className = '';
    document.getElementById('md-explore-info').textContent =
      'n=' + d.n + '，Pearson r=' + d.pearson + '（特征 ' + d.feature + '）';
  }

  async function runExplore() {
    const el = document.getElementById('md-explore-plot');
    document.getElementById('md-explore-info').textContent = '加载中...';
    try {
      const res = await global.API.post('/api/modeling/explore', { feature: curFeature });
      renderExplore(res);
    } catch (e) {
      el.textContent = '探索失败：' + e.message;
    }
  }

  async function exportModelFile() {
    // 问题2-K：下载当前选中/生效模型的 .joblib 文件
    const el = document.getElementById('md-file-status');
    let mid = null;
    try {
      const st = await global.API.get('/api/settings');
      const active = (st.models || []).find(function (m) { return m.is_active; });
      mid = active ? active.id : null;
    } catch (e) { /* ignore */ }
    if (!mid) { el.textContent = '没有生效模型可导出'; return; }
    window.location.href = '/api/models/' + mid + '/export';
    el.textContent = '已触发下载。';
  }

  async function importModelFile(file) {
    const el = document.getElementById('md-file-status');
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await global.API.postForm('/api/models/import_joblib', fd);
      el.textContent = '已导入模型「' + res.name + '」（' + res.type + '）。';
      loadSaved();
    } catch (e) {
      el.textContent = '导入失败：' + e.message;
      el.className = 'status-line err';
    }
  }

  async function predictCsv(file, download) {
    const el = document.getElementById('md-predict-status');
    if (!file) { el.textContent = '请先选择特征 CSV'; return; }
    const fd = new FormData();
    fd.append('file', file);
    try {
      if (download) {
        window.location.href = '#';
        // 直接下载结果 CSV：用隐藏表单提交避免拦截
        const form = document.createElement('form');
        form.method = 'post'; form.action = '/api/modeling/predict_csv/export';
        form.enctype = 'multipart/form-data';
        const inp = document.createElement('input');
        inp.type = 'file'; inp.name = 'file'; inp.hidden = true;
        const dt = new DataTransfer(); dt.items.add(file); inp.files = dt.files;
        form.appendChild(inp);
        document.body.appendChild(form); form.submit(); form.remove();
        el.textContent = '已触发结果 CSV 下载。';
        return;
      }
      const res = await global.API.postForm('/api/modeling/predict_csv', fd);
      const out = document.getElementById('md-predict-out');
      let html = '<table class="md-table"><thead><tr><th>行</th><th>特征值</th><th>浓度 C</th><th>不确定度 U</th><th>判定</th></tr></thead><tbody>';
      (res.predictions || []).forEach(function (p) {
        html += '<tr><td>' + p.row + '</td><td>' + (p.feature_value == null ? '-' : p.feature_value) + '</td>'
          + '<td>' + (p.conc == null ? (p.error || '-') : p.conc) + '</td>'
          + '<td>' + (p.u == null ? '-' : p.u) + '</td>'
          + '<td>' + (p.status || '-') + '</td></tr>';
      });
      html += '</tbody></table>';
      html += '<div class="hint">模型：' + res.model + '，特征列：' + res.feature + '</div>';
      out.innerHTML = html;
      out.className = '';
      el.textContent = '共 ' + res.predictions.length + ' 行。';
    } catch (e) {
      el.textContent = '预测失败：' + e.message;
      el.className = 'status-line err';
    }
  }

  async function runCv() {
    const status = document.getElementById('md-status');
    const box = document.getElementById('md-cv-box');
    status.textContent = '交叉验证中...';
    status.className = 'status-line';
    try {
      await savePreprocess();
      const res = await global.API.post('/api/modeling/cv', {
        feature: curFeature,
        method: document.getElementById('md-cv-method').value,
        k: 5,
      });
      let html = '<table class="feat-table"><thead><tr><th>模型</th><th>R²</th><th>RMSE</th><th>MAE</th><th>折数</th></tr></thead><tbody>';
      const methodNames = { loo: '留一法', kfold: '5 折', leave_group: '留浓度组' };
      Object.keys(res.results).forEach(function (mt) {
        const r = res.results[mt];
        if (r.error) {
          html += '<tr><td><b>' + mt + '</b></td><td colspan="4">' + esc(r.error) + '</td></tr>';
          return;
        }
        const s = r.summary;
        html += '<tr><td><b>' + mt + '</b></td><td>' + s.r2 + '</td><td>' + s.rmse + '</td><td>' + s.mae + '</td><td>' + s.n_folds + '</td></tr>';
      });
      html += '</tbody></table>';
      box.innerHTML = html;
      box.className = '';
      status.textContent = '交叉验证完成（' + (methodNames[res.method] || res.method) + '，' + res.n + ' 个点）';
    } catch (e) {
      status.textContent = '交叉验证失败：' + e.message;
      status.className = 'status-line err';
      box.textContent = '';
    }
  }

  var initialized = false;

  function init() {
    if (initialized) return;
    initialized = true;
    document.getElementById('md-fit').addEventListener('click', runFit);
    document.getElementById('md-save').addEventListener('click', saveModel);
    document.getElementById('md-cv').addEventListener('click', runCv);
    document.getElementById('md-explore').addEventListener('click', runExplore);
    document.getElementById('md-export-file').addEventListener('click', exportModelFile);
    document.getElementById('md-import-file').addEventListener('change', function (e) {
      importModelFile(e.target.files[0]); e.target.value = '';
    });
    document.getElementById('md-predict-csv').addEventListener('change', function (e) {
      predictCsv(e.target.files[0], false); e.target.value = '';
    });
    document.getElementById('md-predict-csv-export').addEventListener('click', function () {
      const inp = document.getElementById('md-predict-csv');
      predictCsv(inp.files && inp.files[0], true);
    });
    document.getElementById('md-feature').addEventListener('change', function () {
      curFeature = document.getElementById('md-feature').value;
      loadData();
    });
    // 恢复预处理设置（问题2-H）
    global.API.get('/api/modeling/preprocess').then(function (p) {
      document.getElementById('md-iqr').checked = p.iqr !== false;
      document.getElementById('md-log').checked = !!p.log_conc;
      document.getElementById('md-zscore').checked = !!p.zscore;
    }).catch(function () { /* 忽略 */ });

    // 问题2-G：CSV/Excel 标定数据导入（预览列 → 选择浓度列/特征列 → 导入）
    const impFile = document.getElementById('md-import-file');
    const impBox = document.getElementById('md-import-box');
    const impStatus = document.getElementById('md-imp-status');
    document.getElementById('md-import').addEventListener('click', function () { impFile.click(); });
    impFile.addEventListener('change', async function () {
      if (!impFile.files.length) return;
      const fd = new FormData();
      fd.append('file', impFile.files[0]);
      impStatus.textContent = '解析文件中...';
      impStatus.className = 'status-line';
      try {
        const res = await global.API.postForm('/api/modeling/import_preview', fd);
        const concSel = document.getElementById('md-imp-conc');
        const featSel = document.getElementById('md-imp-feat');
        concSel.innerHTML = '';
        featSel.innerHTML = '';
        res.columns.forEach(function (c) {
          concSel.appendChild(new Option(c, c));
          featSel.appendChild(new Option(c, c));
        });
        const guess = res.columns.find(function (c) { return /conc|浓度/i.test(c); });
        const fguess = res.columns.find(function (c) { return /(_R$|_G$|_B$|mean|hue|T_R|Bg|ratio|od_|OD)/i.test(c); });
        if (guess) concSel.value = guess;
        if (fguess) featSel.value = fguess;
        document.getElementById('md-imp-preview').textContent =
          '列：' + res.columns.join('、') + '　共 ' + res.total_rows + ' 行，预览前 ' + Math.min(5, res.preview.length) + ' 行。';
        impBox.style.display = 'block';
        impStatus.textContent = '';
      } catch (e) {
        impStatus.textContent = '解析失败：' + e.message;
        impStatus.className = 'status-line err';
      }
    });
    document.getElementById('md-import-go').addEventListener('click', async function () {
      if (!impFile.files.length) return;
      const fd = new FormData();
      fd.append('file', impFile.files[0]);
      fd.append('conc_col', document.getElementById('md-imp-conc').value);
      fd.append('feature_col', document.getElementById('md-imp-feat').value);
      const impUnit = document.getElementById('md-imp-unit');
      fd.append('unit', impUnit ? impUnit.value : 'ng/mL');
      impStatus.textContent = '导入中...';
      impStatus.className = 'status-line';
      try {
        const res = await global.API.postForm('/api/modeling/import', fd);
        impStatus.textContent = '导入完成：' + res.imported + ' 个数据点（' + res.groups + ' 个浓度组）'
          + (res.skipped ? '，跳过 ' + res.skipped + ' 行' : '');
        impStatus.className = 'status-line';
        await loadData();
      } catch (e) {
        impStatus.textContent = '导入失败：' + e.message;
        impStatus.className = 'status-line err';
      }
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

