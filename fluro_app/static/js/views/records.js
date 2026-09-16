/* views/records.js：检测记录（M7）检索 / 详情 / 导出 */
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

  async function load() {
    const q = document.getElementById('rec-q').value.trim();
    const batch = document.getElementById('rec-batch').value.trim();
    const params = new URLSearchParams();
    if (q) params.set('q', q);
    if (batch) params.set('batch', batch);
    const status = document.getElementById('rec-status');
    status.textContent = '加载中...';
    try {
      const res = await global.API.get('/api/detections?' + params.toString());
      render(res.detections);
      status.textContent = res.detections.length + ' 条记录';
    } catch (e) {
      status.textContent = '加载失败：' + e.message;
    }
  }

  function render(rows) {
    const el = document.getElementById('rec-list');
    if (!rows.length) {
      el.innerHTML = '<div class="empty">没有匹配的记录。</div>';
      return;
    }
    let html = '<table class="md-table"><thead><tr>'
      + '<th>ID</th><th>图片</th><th>批次</th><th>浓度 C</th><th>U(95%)</th>'
      + '<th>区间</th><th>判定</th><th>时间</th><th>操作</th></tr></thead><tbody>';
    rows.forEach(function (r) {
      const jm = judgeMap[r.status] || [r.status, 'st-uploaded'];
      const lo = (r.conc - r.u).toFixed(2), hi = (r.conc + r.u).toFixed(2);
      html += '<tr><td>' + r.id + '</td><td>#' + r.image_id + '</td><td>' + esc(r.image_batch || '-') + '</td>'
        + '<td><b>' + Number(r.conc).toFixed(2) + '</b></td><td>±' + Number(r.u).toFixed(2) + '</td>'
        + '<td>[' + lo + ', ' + hi + ']</td>'
        + '<td><span class="gstatus ' + jm[1] + '">' + jm[0] + '</span></td>'
        + '<td>' + esc((r.created_at || '').slice(0, 19)) + '</td>'
        + '<td><button class="btn small" data-detail="' + r.id + '">详情</button></td></tr>';
    });
    html += '</tbody></table>';
    el.innerHTML = html;
    el.querySelectorAll('[data-detail]').forEach(function (b) {
      b.addEventListener('click', function () { showDetail(parseInt(b.dataset.detail, 10)); });
    });
  }

  async function showDetail(detId) {
    // 用当前加载的记录找对应行（详情数据在快照中）
    const rowsEl = document.querySelectorAll('#rec-list tr');
    const res = await global.API.get('/api/detections');
    const row = res.detections.find(function (d) { return d.id === detId; });
    if (!row) return;
    let snap;
    try { snap = JSON.parse(row.params_snapshot_json || '{}'); } catch (e) { snap = {}; }
    let html = '<div class="det-card" style="margin-top:10px;"><div class="det-main">'
      + '<div class="det-conc" style="font-size:22px;">' + Number(row.conc).toFixed(2) + ' <span class="det-unit">ng/mL</span></div>'
      + '<div class="det-u">U(95%) = ±' + Number(row.u).toFixed(2) + '</div></div>'
      + '<div class="det-meta">'
      + '<div>模型：' + esc(snap.model_name || '-') + '（' + esc(snap.model_type || '') + '，R²=' + ((snap.model_metrics && snap.model_metrics.r2) != null ? Number(snap.model_metrics.r2).toFixed(4) : '-') + '）</div>'
      + '<div>模型参数：' + esc(JSON.stringify(snap.model_params || {})) + '</div>'
      + '<div>特征：' + esc(snap.feature || '-') + ' = ' + (snap.feature_value != null ? Number(snap.feature_value).toFixed(3) : '-') + '</div>'
      + '<div>限值：' + ((snap.limits && snap.limits.lower) == null ? '无' : snap.limits.lower) + ' ~ ' + ((snap.limits && snap.limits.upper) == null ? '无' : snap.limits.upper) + '</div>'
      + '<div>标定点数：' + (snap.calibration_n != null ? snap.calibration_n : '-') + '　置信度：' + (snap.conf != null ? snap.conf : '-') + '</div>'
      + '</div></div>';
    const container = document.getElementById('rec-list');
    const old = container.querySelector('.det-card');
    if (old) old.remove();
    container.insertAdjacentHTML('beforeend', html);
  }

  function exportCsv() {
    const q = document.getElementById('rec-q').value.trim();
    const batch = document.getElementById('rec-batch').value.trim();
    const params = new URLSearchParams();
    if (q) params.set('q', q);
    if (batch) params.set('batch', batch);
    window.location.href = '/api/detections/export?' + params.toString();
  }

  var initialized = false;

  function init() {
    if (initialized) return;
    initialized = true;
    document.getElementById('rec-search').addEventListener('click', load);
    document.getElementById('rec-export').addEventListener('click', exportCsv);
    document.getElementById('rec-q').addEventListener('keydown', function (e) { if (e.key === 'Enter') load(); });
    document.getElementById('rec-batch').addEventListener('keydown', function (e) { if (e.key === 'Enter') load(); });
    load();
  }

  function onView() { init(); }

  global.FluroRecords = { init: init, onView: onView };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
