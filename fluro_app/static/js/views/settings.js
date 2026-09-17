/* views/settings.js：数据与设置（M8）限值/参考图/模板/模型/备份恢复 */
(function (global) {
  'use strict';

  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function fmtSize(n) {
    if (n == null) return '-';
    if (n > 1048576) return (n / 1048576).toFixed(1) + ' MB';
    if (n > 1024) return (n / 1024).toFixed(1) + ' KB';
    return n + ' B';
  }

  async function load() {
    try {
      const res = await global.API.get('/api/settings');
      const st = res.settings || {};
      document.getElementById('st-lower').value = st.limit_lower || '';
      document.getElementById('st-upper').value = st.limit_upper || '';
      document.getElementById('st-unit').value = st.unit || 'ng/mL';
      document.getElementById('st-batch').value = st.current_batch || '';
      document.getElementById('st-dir').textContent = res.data_dir || '-';
      renderRefs(res.refs);
      renderTemplates(res.templates);
      renderModels(res.models);
      renderBackups(res.backups);
      document.getElementById('st-status').textContent = '';
    } catch (e) {
      document.getElementById('st-status').textContent = '加载失败：' + e.message;
    }
  }

  async function loadExperiment() {
    try {
      const ex = await global.API.get('/api/experiment');
      document.getElementById('ex-analyte').value = ex.analyte || '';
      document.getElementById('ex-trend').value = ex.color_trend || '';
      document.getElementById('ex-carrier').value = ex.carrier || 'single_card';
      document.getElementById('ex-roi-count').value = ex.roi_count != null ? ex.roi_count : 2;
      document.getElementById('ex-roi-names').value = (ex.roi_names || []).join(',');
      document.getElementById('ex-colorcard').checked = !!ex.has_color_card;
      document.getElementById('ex-blankref').checked = ex.has_blank_ref !== false;
      document.getElementById('ex-concs').value = ex.known_concs || '';
      document.getElementById('ex-replicates').value = ex.replicates || '';
      document.getElementById('ex-images').value = ex.image_count || '';
      document.getElementById('ex-note').value = ex.note || '';
    } catch (e) { /* 忽略 */ }
  }

  async function saveExperiment() {
    const status = document.getElementById('ex-status');
    const names = document.getElementById('ex-roi-names').value.split(',').map(function (s) { return s.trim(); }).filter(Boolean);
    const payload = {
      analyte: document.getElementById('ex-analyte').value,
      color_trend: document.getElementById('ex-trend').value,
      carrier: document.getElementById('ex-carrier').value,
      roi_count: parseInt(document.getElementById('ex-roi-count').value, 10),
      roi_names: names,
      has_color_card: document.getElementById('ex-colorcard').checked,
      has_blank_ref: document.getElementById('ex-blankref').checked,
      known_concs: document.getElementById('ex-concs').value,
      replicates: document.getElementById('ex-replicates').value,
      image_count: document.getElementById('ex-images').value,
      note: document.getElementById('ex-note').value,
    };
    status.textContent = '保存中...';
    try {
      const res = await global.API.put('/api/experiment', payload);
      status.textContent = '实验信息已保存（ROI：' + res.experiment.roi_names.join('、') + '）';
      status.className = 'status-line';
    } catch (e) {
      status.textContent = '保存失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  function renderRefs(refs) {
    const el = document.getElementById('st-refs');
    const dark = refs && refs.dark_path ? refs.dark_path : '未设置';
    const flat = refs && refs.flat_path ? refs.flat_path : '未设置';
    el.innerHTML =
      '<div class="det-meta">暗场参考图：' + esc(dark) + '</div>'
      + '<div class="det-meta">平场参考图：' + esc(flat) + '</div>'
      + '<div class="hint">在「检测工作流」第 2 步可上传参考图。</div>';
  }

  function renderTemplates(templates) {
    const el = document.getElementById('st-templates');
    if (!templates || !templates.length) {
      el.innerHTML = '<div class="empty">尚无 ROI 模板。</div>';
      return;
    }
    let html = '<table class="md-table"><thead><tr><th>名称</th><th>位置</th><th>状态</th><th>操作</th></tr></thead><tbody>';
    templates.forEach(function (t) {
      html += '<tr><td>' + esc(t.name) + '</td><td>(' + t.x + ', ' + t.y + ', ' + t.w + ', ' + t.h + ')</td>'
        + '<td>' + (t.is_active ? '<span class="pb pb-ok">激活</span>' : '') + '</td>'
        + '<td><button class="btn small" data-act="' + t.id + '">设为激活</button> '
        + '<button class="btn small" data-del="' + t.id + '">删除</button></td></tr>';
    });
    html += '</tbody></table>';
    el.innerHTML = html;
    el.querySelectorAll('[data-act]').forEach(function (b) {
      b.addEventListener('click', async function () {
        await global.API.post('/api/templates/' + b.dataset.act + '/activate', {});
        load();
      });
    });
    el.querySelectorAll('[data-del]').forEach(function (b) {
      b.addEventListener('click', async function () {
        if (!confirm('删除该模板？')) return;
        await global.API.post('/api/templates/' + b.dataset.del + '/delete', {});
        load();
      });
    });
  }

  function renderModels(models) {
    const el = document.getElementById('st-models');
    if (!models || !models.length) {
      el.innerHTML = '<div class="empty">尚无模型。</div>';
      return;
    }
    let html = '<table class="md-table"><thead><tr><th>名称</th><th>类型</th><th>状态</th><th>操作</th></tr></thead><tbody>';
    models.forEach(function (m) {
      html += '<tr><td>' + esc(m.name) + '</td><td>' + esc(m.type) + '</td>'
        + '<td>' + (m.is_active ? '<span class="pb pb-ok">生效中</span>' : '') + '</td>'
        + '<td><button class="btn small" data-act="' + m.id + '">设为生效</button> '
        + '<button class="btn small" data-del="' + m.id + '">删除</button></td></tr>';
    });
    html += '</tbody></table>';
    el.innerHTML = html;
    el.querySelectorAll('[data-act]').forEach(function (b) {
      b.addEventListener('click', async function () {
        await global.API.post('/api/models/' + b.dataset.act + '/activate', {});
        load();
      });
    });
    el.querySelectorAll('[data-del]').forEach(function (b) {
      b.addEventListener('click', async function () {
        if (!confirm('删除该模型？')) return;
        await global.API.post('/api/models/' + b.dataset.del + '/delete', {});
        load();
      });
    });
  }

  function renderBackups(backups) {
    const el = document.getElementById('st-backups');
    if (!backups || !backups.length) {
      el.innerHTML = '<div class="empty">暂无备份。建议定期「立即备份」。</div>';
      return;
    }
    let html = '<table class="md-table"><thead><tr><th>备份</th><th>大小</th><th>操作</th></tr></thead><tbody>';
    backups.forEach(function (b) {
      html += '<tr><td>' + esc(b.name) + '</td><td>' + fmtSize(b.size) + '</td>'
        + '<td><button class="btn small" data-restore="' + esc(b.name) + '">恢复</button> '
        + '<button class="btn small" data-del="' + esc(b.name) + '">删除</button></td></tr>';
    });
    html += '</tbody></table>';
    el.innerHTML = html;
    el.querySelectorAll('[data-restore]').forEach(function (b) {
      b.addEventListener('click', async function () {
        if (!confirm('恢复将覆盖当前全部数据（恢复前会自动备份当前数据）。确认继续？')) return;
        try {
          const r = await global.API.post('/api/backup/restore', { name: b.dataset.restore });
          document.getElementById('st-status').textContent = '已恢复：' + r.restored + '（安全备份：' + r.safe_backup + '）';
          await load();
        } catch (e) {
          document.getElementById('st-status').textContent = '恢复失败：' + e.message;
        }
      });
    });
    el.querySelectorAll('[data-del]').forEach(function (b) {
      b.addEventListener('click', async function () {
        if (!confirm('删除备份 ' + b.dataset.del + '？')) return;
        await global.API.post('/api/backup/' + b.dataset.del + '/delete', {});
        load();
      });
    });
  }

  async function saveSettings() {
    const lower = document.getElementById('st-lower').value;
    const upper = document.getElementById('st-upper').value;
    const unit = document.getElementById('st-unit').value;
    const batch = document.getElementById('st-batch').value;
    const st = {};
    st.limit_lower = lower === '' ? null : lower;
    st.limit_upper = upper === '' ? null : upper;
    if (unit) st.unit = unit;
    if (batch) st.current_batch = batch;
    try {
      await global.API.post('/api/settings', { settings: st });
      document.getElementById('st-status').textContent = '设置已保存';
      if (global.FluroApp) global.FluroApp.refreshTopbar();
    } catch (e) {
      document.getElementById('st-status').textContent = '保存失败：' + e.message;
    }
  }

  async function doBackup() {
    const label = prompt('备份备注（可选）：', '');
    if (label === null) return;
    try {
      const r = await global.API.post('/api/backup', { label: label || undefined });
      document.getElementById('st-status').textContent = '备份完成：' + r.backup.name + '（' + fmtSize(r.backup.size) + '）';
      await load();
    } catch (e) {
      document.getElementById('st-status').textContent = '备份失败：' + e.message;
    }
  }

  var initialized = false;

  function init() {
    if (initialized) return;
    initialized = true;
    document.getElementById('st-save').addEventListener('click', saveSettings);
    document.getElementById('st-backup').addEventListener('click', doBackup);
    document.getElementById('st-refresh').addEventListener('click', load);
    document.getElementById('ex-save').addEventListener('click', saveExperiment);
    load();
    loadExperiment();
  }

  function onView() { if (!initialized) init(); else load(); }

  global.FluroSettings = { init: init, onView: onView };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
