/* app.js：导航切换、全局状态条、通用工具 */
(function (global) {
  'use strict';

  function switchView(name) {
    document.querySelectorAll('.nav-item').forEach(function (n) {
      n.classList.toggle('active', n.getAttribute('data-view') === name);
    });
    document.querySelectorAll('.view').forEach(function (v) {
      v.classList.toggle('active', v.id === 'view-' + name);
    });
    if (global.FluroHome && typeof global.FluroHome.onView === 'function') {
      global.FluroHome.onView(name);
    }
    if (global.FluroWorkflow && typeof global.FluroWorkflow.onView === 'function') {
      global.FluroWorkflow.onView(name);
    }
    if (name === 'modeling' && global.FluroModeling && typeof global.FluroModeling.onView === 'function') {
      global.FluroModeling.onView();
    }
    if (name === 'records' && global.FluroRecords && typeof global.FluroRecords.onView === 'function') {
      global.FluroRecords.onView();
    }
    if (name === 'settings' && global.FluroSettings && typeof global.FluroSettings.onView === 'function') {
      global.FluroSettings.onView();
    }
  }

  document.querySelectorAll('.nav-item[data-view]').forEach(function (n) {
    n.addEventListener('click', function () { switchView(n.getAttribute('data-view')); });
  });

  async function refreshTopbar() {
    try {
      const s = await global.API.get('/api/home/summary');
      document.getElementById('ts-model').textContent = s.model_name || '-';
      document.getElementById('ts-r2').textContent = (s.model_r2 == null) ? '-' : Number(s.model_r2).toFixed(4);
      document.getElementById('ts-batch').textContent = s.current_batch || '-';
      document.getElementById('ts-dir').textContent = s.data_dir || '-';
      document.getElementById('ts-today').textContent = s.today_count == null ? '-' : s.today_count;
    } catch (e) {
      // 首页聚合接口在 M7 接入；此前的阶段保持占位
    }
  }

  global.FluroApp = {
    switchView: switchView,
    refreshTopbar: refreshTopbar,
    el: function (id) { return document.getElementById(id); },
  };

  document.addEventListener('DOMContentLoaded', function () {
    refreshTopbar();
    // 切换用途时显示/隐藏已知浓度输入
    const kind = document.getElementById('up-kind');
    const concWrap = document.getElementById('up-conc-wrap');
    if (kind && concWrap) {
      kind.addEventListener('change', function () {
        concWrap.style.display = kind.value === 'calibration' ? 'inline-flex' : 'none';
      });
    }
  });
})(window);
