/* views/workflow.js：检测工作流视图
   M1：上传与缩略图墙；M2：预处理；M3：ROI 与特征；M4+ 流水线 */
(function (global) {
  'use strict';

  var currentImageId = null;
  var roiEditor = null;

  var STATUS_LABEL = {
    ok: '完成', attention: '需人工', error: '出错', uploaded: '待处理', pending: '待处理',
  };

  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function renderGallery(images) {
    const g = document.getElementById('up-gallery');
    g.innerHTML = '';
    images.forEach(function (im) {
      const div = document.createElement('div');
      div.className = 'gitem' + (im.id === currentImageId ? ' selected' : '');
      div.dataset.id = im.id;
      const del = document.createElement('button');
      del.className = 'gdel';
      del.title = '删除该图及其记录';
      del.textContent = '×';
      del.addEventListener('click', async function (ev) {
        ev.stopPropagation();
        if (!confirm('删除该图及其全部记录与文件？')) return;
        try {
          await global.API.del('/api/images/' + im.id);
          if (currentImageId === im.id) currentImageId = null;
          await refreshGallery();
          if (global.FluroApp) global.FluroApp.refreshTopbar();
        } catch (e) {
          alert('删除失败：' + e.message);
        }
      });
      const img = document.createElement('img');
      img.src = im.thumb_url;
      img.alt = '缩略图';
      const meta = document.createElement('div');
      meta.className = 'gmeta';
      const kindTag = document.createElement('span');
      kindTag.className = 'gkind ' + im.kind;
      kindTag.textContent = im.kind === 'calibration' ? '标定' : '检测';
      meta.appendChild(kindTag);
      const stTag = document.createElement('span');
      stTag.className = 'gstatus st-' + (im.status || 'uploaded');
      stTag.textContent = STATUS_LABEL[im.status] || im.status || '待处理';
      meta.appendChild(stTag);
      const info = document.createElement('div');
      const nameParts = [];
      if (im.filename) nameParts.push(im.filename);
      if (im.known_conc != null) nameParts.push('浓度 ' + im.known_conc + (im.conc_unit ? ' ' + im.conc_unit : ''));
      else if (im.batch) nameParts.push('批次 ' + im.batch);
      info.textContent = nameParts.join(' · ');
      meta.appendChild(info);
      div.appendChild(del);
      div.appendChild(img);
      div.appendChild(meta);
      div.addEventListener('click', function () { selectImage(im.id); });
      g.appendChild(div);
    });
  }

  async function refreshGallery() {
    try {
      const res = await global.API.get('/api/images');
      renderGallery(res.images);
    } catch (e) { /* 忽略 */ }
  }

  function renderPipeStatus(results) {
    const el = document.getElementById('pipe-status');
    const okN = results.filter(function (r) { return r.status === 'ok'; }).length;
    const attN = results.filter(function (r) { return r.status === 'attention'; }).length;
    const errN = results.filter(function (r) { return r.status === 'error'; }).length;
    let html = '<span class="pb pb-ok">自动完成 ' + okN + ' 张</span>';
    if (attN) html += '<span class="pb pb-attention">需人工处理 ' + attN + ' 张</span>';
    if (errN) html += '<span class="pb pb-error">出错 ' + errN + ' 张</span>';
    const attList = results.filter(function (r) { return r.status === 'attention'; })
      .map(function (r) { return '<a href="javascript:void(0)" data-id="' + r.image_id + '" class="att-link">#' + r.image_id + '</a>'; })
      .join(' ');
    if (attList) html += '<div>需人工：' + attList + '（点击进入该图修正）</div>';
    el.innerHTML = html;
    el.querySelectorAll('.att-link').forEach(function (a) {
      a.addEventListener('click', function () { selectImage(parseInt(a.dataset.id, 10)); });
    });
  }

  function markThumbSelected(id) {
    currentImageId = id;
    document.querySelectorAll('#up-gallery .gitem').forEach(function (n) {
      n.classList.toggle('selected', n.dataset.id === String(id));
    });
  }

  async function runPipeline(ids) {
    const el = document.getElementById('pipe-status');
    el.innerHTML = '<span class="pb pb-uploaded">自动流水线执行中...</span>';
    try {
      const res = await global.API.post('/api/pipeline/run', { image_ids: ids });
      renderPipeStatus(res.results);
      await refreshGallery();
      if (global.FluroApp) global.FluroApp.refreshTopbar();
      // 问题1（第四批）：流水线状态变化后实时刷新首页工作流卡片
      if (global.FluroHome && typeof global.FluroHome.refresh === 'function') {
        global.FluroHome.refresh();
      }
      // 问题2：自动处理全部后直接跳到结果页显示数据
      const okImgs = (res.results || []).filter(function (r) {
        return r.steps && r.steps.result === 'ok';
      });
      if (okImgs.length) {
        markThumbSelected(okImgs[0].image_id);
        showWfPanel(6);
        await loadResultPanel();
        el.innerHTML = '<span class="pb pb-ok">自动流水线完成：' + okImgs.length + ' 张图已产出检测结果（已显示第一张）。</span>';
      } else if (res.results && res.results.length) {
        const hasAttention = res.results.some(function (r) {
          return r.steps && r.steps.result === 'attention';
        });
        el.innerHTML = hasAttention
          ? '<span class="pb pb-attention">流水线已跑完，但尚未保存生效标定模型，无法产出检测结果。请先到「标定建模」页拟合并保存模型，再回来重试。</span>'
          : '<span class="pb pb-uploaded">流水线已跑完（标定图无需检测）。</span>';
      }
    } catch (e) {
      el.innerHTML = '<span class="pb pb-error">流水线执行失败：' + e.message + '</span>';
    }
  }

  function selectImage(id) {
    currentImageId = id;
    document.querySelectorAll('#up-gallery .gitem').forEach(function (n) {
      n.classList.toggle('selected', n.dataset.id === String(id));
    });
    // 显示预处理面板并自动用默认参数重跑
    showWfPanel(2);
    document.getElementById('pp-status').textContent = '已选中图片 #' + id + '，自动执行预处理...';
    runPreprocess(true);
  }

  // 问题1（第四批）：首页工作流卡片点击某步图像 → 跳到工作流对应步骤并选中该图
  function openStep(id, step) {
    markThumbSelected(id);
    if (step === 1) { showWfPanel(1); return; }
    showWfPanel(step);                 // 3/4/5/6 面板会自动加载该图数据
    if (step === 2) runPreprocess(true); // 预处理面板：自动重跑并显示该图最新处理图
  }

  function showWfPanel(n) {
    document.querySelectorAll('.wf-panel').forEach(function (p) { p.style.display = 'none'; });
    document.getElementById('wf-panel-' + n).style.display = 'block';
    document.querySelectorAll('.wf-step').forEach(function (s) {
      s.classList.toggle('active', s.dataset.wf === String(n));
      s.classList.toggle('done', parseInt(s.dataset.wf, 10) < n);
    });
    if (n === 3) loadChannelsPanel();
    if (n === 4) loadRoiPanel();
    if (n === 5) refreshFeatures();
    if (n === 6) loadResultPanel();
  }

  // ---- M6：检测结果 ----

  var detInited = false;

  async function loadResultPanel() {
    if (!currentImageId) {
      document.getElementById('det-zone').innerHTML = '<div class="empty">请先在上方选择一张图片。</div>';
      return;
    }
    try {
      const imRes = await global.API.get('/api/images');
      const im = imRes.images.find(function (i) { return i.id === currentImageId; });
      const zone = document.getElementById('det-zone');
      const status = document.getElementById('det-status');
      // 处理后图（ROI 标注）
      const detImg = document.getElementById('det-overlay');
      if (detImg) {
        detImg.src = '/api/images/' + currentImageId + '/overlay?t=' + Date.now();
        if (detImg.closest('div')) detImg.closest('div').style.display = '';
      }
      if (im && im.kind === 'calibration') {
        const cu = im.conc_unit || 'ng/mL';
        zone.innerHTML = '<div class="card">'
          + '<h3>标定数据录入（写浓度 → 建模型）</h3>'
          + '<p class="hint">把这张标定图写入已知浓度并加入标定数据集；再到「标定建模」页拟合并保存生效模型。</p>'
          + '<div class="form-row">'
          + '<label>浓度（' + esc(cu) + '）：<input id="cal-conc" type="number" step="any" value="' + (im.known_conc != null ? im.known_conc : '') + '"></label>'
          + '<button id="cal-add" class="btn primary">加入标定数据集</button>'
          + '<span id="cal-msg" class="status-line" style="margin:0;"></span>'
          + '</div><div id="cal-state"></div></div>';
        document.getElementById('cal-add').addEventListener('click', quickCalibrate);
        loadCalState(currentImageId);
        return;
      }
      // 已检测过则展示历史
      const det = await global.API.get('/api/detections');
      const row = det.detections.find(function (d) { return d.image_id === currentImageId; });
      if (row) {
        renderDetection({ detection: row, from_history: true });
      } else {
        zone.innerHTML = '<div class="empty">尚未检测。点击下方按钮使用生效模型执行检测。</div>';
        status.textContent = '';
      }
      if (!detInited) {
        detInited = true;
        const btn = document.getElementById('det-run');
        if (btn) btn.addEventListener('click', runDetect);
        const bm = document.getElementById('det-build-model');
        if (bm) bm.addEventListener('click', buildModelNow);
      }
    } catch (e) { /* 忽略 */ }
  }

  async function quickCalibrate() {
    const concInput = document.getElementById('cal-conc');
    if (!concInput || !currentImageId) return;
    const conc = concInput.value;
    const msg = document.getElementById('cal-msg');
    if (!conc || isNaN(parseFloat(conc))) {
      msg.textContent = '请输入有效浓度';
      msg.className = 'status-line err';
      return;
    }
    msg.textContent = '加入中...';
    try {
      const res = await global.API.post('/api/calibration/quick', { image_id: currentImageId, conc: parseFloat(conc) });
      msg.textContent = '已加入浓度 ' + res.conc + ' ' + (res.unit || 'ng/mL')
        + (res.reused ? '（更新已有数据点）' : '') + '，请到「标定建模」页拟合并保存模型';
      msg.className = 'status-line';
      loadCalState(currentImageId);
    } catch (e) {
      msg.textContent = '加入失败：' + e.message;
      msg.className = 'status-line err';
    }
  }

  async function loadCalState(imageId) {
    const el = document.getElementById('cal-state');
    if (!el) return;
    try {
      const data = await global.API.get('/api/calibration/data');
      const joined = [];
      (data.groups || []).forEach(function (g) {
        (g.points || []).forEach(function (p) {
          if (p.image_id === imageId) joined.push(g.conc);
        });
      });
      el.innerHTML = joined.length
        ? '<div class="hint">该图已加入浓度分组：' + joined.join('、') + '。可在「标定建模」页拟合并保存生效模型。</div>'
        : '<div class="hint">尚未加入分组：填写浓度后点击「加入标定数据集」。</div>';
    } catch (e) { el.innerHTML = ''; }
  }

  // 问题3：结果页一键生成模型（标定数据 → 拟合 → 保存生效）
  async function buildModelNow() {
    const msg = document.getElementById('det-build-msg');
    if (!msg) return;
    msg.textContent = '生成模型中...';
    msg.className = 'status-line';
    try {
      const data = await global.API.get('/api/calibration/data');
      const n = (data.groups || []).reduce(function (s, g) {
        return s + (g.points || []).length;
      }, 0);
      if (!n || n < 3) {
        msg.textContent = '标定数据不足（至少 3 个不同浓度）。请先在「标定建模」页导入标定数据，或在标定图结果页录入浓度。';
        msg.className = 'status-line err';
        return;
      }
      // 优先沿用当前生效模型的特征，否则用标准导入特征 T_R_over_Bg_R
      let feature = 'T_R_over_Bg_R';
      try {
        const ms = await global.API.get('/api/models');
        const act = (ms.models || []).find(function (m) { return m.is_active; });
        if (act && act.feature) feature = act.feature;
      } catch (e) { /* 无模型时用默认特征 */ }
      const fit = await global.API.post('/api/calibration/fit', { feature: feature });
      if (!fit.ok || !fit.best || !fit.results[fit.best]) {
        throw new Error(fit.error || '拟合失败');
      }
      const best = fit.results[fit.best];
      const name = '工作流生成 ' + new Date().toLocaleString();
      const saved = await global.API.post('/api/models', {
        name: name, type: fit.best, params: best.params,
        metrics: { r2: best.r2, rmse: best.rmse, lod: best.lod },
        source_snapshot: { feature: fit.feature, data: fit.data, n: fit.n,
                           unit: fit.unit || 'ng/mL', preprocess: fit.preprocess },
      });
      msg.textContent = '模型已生成并生效：' + name + '（' + fit.best + '，R²=' + Number(best.r2).toFixed(4)
        + (best.lod == null ? '' : '，LOD=' + Number(best.lod).toFixed(3)) + '）';
      msg.className = 'status-line';
      if (global.FluroApp) global.FluroApp.refreshTopbar();
      // 用新模型重新检测当前图并刷新结果卡
      if (currentImageId) {
        try {
          const res = await global.API.post('/api/detect/' + currentImageId, {});
          renderDetection(res);
        } catch (e2) { /* 保留原结果 */ }
      }
    } catch (e) {
      msg.textContent = '生成失败：' + e.message;
      msg.className = 'status-line err';
    }
  }

  async function runDetect() {
    if (!currentImageId) return;
    const status = document.getElementById('det-status');
    status.textContent = '检测中...';
    try {
      const res = await global.API.post('/api/detect/' + currentImageId, {});
      renderDetection(res);
      status.textContent = '检测完成，已保存记录 #' + res.detection.id + '（' + new Date().toLocaleString() + '）';
      status.className = 'status-line';
      if (global.FluroApp) global.FluroApp.refreshTopbar();
    } catch (e) {
      status.textContent = '检测失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  function renderDetection(res) {
    const d = res.detection;
    const zone = document.getElementById('det-zone');
    const judgeMap = {
      within: ['正常范围', 'st-ok'],
      above: ['高于上限', 'st-error'],
      below: ['低于下限', 'st-error'],
      borderline: ['临界（C±U 跨限）', 'st-attention'],
    };
    const jm = judgeMap[d.status] || [d.status, 'st-uploaded'];
    const lim = d.limits || {};
    const unit = d.unit || 'ng/mL';
    zone.innerHTML =
      '<div class="det-card"><div class="det-main">'
      + '<div class="det-conc">' + Number(d.conc).toFixed(2) + ' <span class="det-unit">' + esc(unit) + '</span></div>'
      + '<div class="det-u">U(95%) = ±' + Number(d.u).toFixed(2) + ' &nbsp; 区间 [' + (d.conc - d.u).toFixed(2) + ', ' + (d.conc + d.u).toFixed(2) + ']</div>'
      + '<div class="det-judge"><span class="gstatus ' + jm[1] + '">' + jm[0] + '</span></div>'
      + '</div><div class="det-meta">'
      + '<div>模型：' + esc(d.model_name || '-') + '（R²=' + (d.model_r2 == null ? '-' : Number(d.model_r2).toFixed(4)) + '）</div>'
      + '<div>特征：' + esc(d.feature || '-') + ' = ' + Number(d.feature_value).toFixed(3) + '</div>'
      + '<div>限值：' + (lim.lower == null ? '无' : lim.lower) + ' ~ ' + (lim.upper == null ? '无' : lim.upper) + '</div>'
      + (d.from_history ? '<div class="hint">该结果为历史记录（ROI/预处理变更会自动重算）</div>' : '')
      + '</div></div>'
      + '<div class="det-actions">'
      + '<button id="det-save" class="btn">' + (d.id ? '保存结果（更新记录 #' + d.id + '）' : '保存结果') + '</button>'
      + '<button id="det-run" class="btn primary">重新检测</button>'
      + '</div>';
    document.getElementById('det-save').addEventListener('click', saveDetectResult);
    document.getElementById('det-run').addEventListener('click', runDetect);
  }

  async function saveDetectResult() {
    if (!currentImageId) return;
    const status = document.getElementById('det-status');
    status.textContent = '保存中...';
    try {
      const res = await global.API.post('/api/detect/' + currentImageId, {});
      renderDetection(res);
      status.textContent = '已保存到检测记录 #' + res.detection.id + '（' + new Date().toLocaleString() + '）';
      status.className = 'status-line';
      if (global.FluroApp) global.FluroApp.refreshTopbar();
    } catch (e) {
      status.textContent = '保存失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  // ---- M3：ROI ----

  function ensureRoiEditor() {
    if (!roiEditor) {
      roiEditor = global.ROIEditor.create(document.getElementById('roi-canvas'), {
        onRoiChange: function () { /* 拖动实时反馈；保存时提交 */ },
      });
    }
    return roiEditor;
  }

  async function loadTemplates(selectedId) {
    try {
      const res = await global.API.get('/api/templates');
      const sel = document.getElementById('roi-tpl-select');
      sel.innerHTML = '';
      res.templates.forEach(function (t) {
        const opt = document.createElement('option');
        opt.value = t.id;
        opt.textContent = t.name + (t.is_active ? '（激活）' : '');
        if (t.is_active || t.id === selectedId) opt.selected = true;
        sel.appendChild(opt);
      });
    } catch (e) { /* 忽略 */ }
  }

  // ---- 问题5：RGB 通道分离 ----
  async function loadChannelsPanel() {
    if (!currentImageId) return;
    const status = document.getElementById('ch-status');
    const zone = document.getElementById('ch-zone');
    try {
      const res = await global.API.get('/api/pipeline/' + currentImageId + '/channels');
      if (!res.channels || !res.channels.r_url) {
        zone.innerHTML = '<div class="empty">尚无通道数据：点击「重新分离通道」生成（基于处理后图像）。</div>';
        status.textContent = '';
        return;
      }
      const c = res.channels;
      const t = Date.now();
      zone.innerHTML =
        '<div class="ch-grid">'
        + '<figure><img src="' + c.r_url + '?t=' + t + '" alt="R 通道"><figcaption>R 通道（均值 ' + c.mean_r + '）</figcaption></figure>'
        + '<figure><img src="' + c.g_url + '?t=' + t + '" alt="G 通道"><figcaption>G 通道（均值 ' + c.mean_g + '）</figcaption></figure>'
        + '<figure><img src="' + c.b_url + '?t=' + t + '" alt="B 通道"><figcaption>B 通道（均值 ' + c.mean_b + '）</figcaption></figure>'
        + '</div>'
        + '<div class="hint">通道均值与「特征提取」中的平均 R/G/B 对应（特征基于 ROI 区域，此处为全图均值）。</div>';
      status.textContent = '通道已分离（基于处理后图像）';
      status.className = 'status-line';
    } catch (e) {
      status.textContent = '通道数据读取失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  async function runChannels() {
    if (!currentImageId) {
      document.getElementById('ch-status').textContent = '请先选中一张图';
      return;
    }
    const status = document.getElementById('ch-status');
    status.textContent = '分离中...';
    try {
      await global.API.post('/api/pipeline/' + currentImageId + '/channels', {});
      await loadChannelsPanel();
    } catch (e) {
      status.textContent = '通道分离失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  async function loadRoiPanel() {
    if (!currentImageId) return;
    const status = document.getElementById('roi-status');
    status.textContent = '加载中...';
    ensureRoiEditor();
    try {
      const roiRes = await global.API.get('/api/pipeline/' + currentImageId + '/roi');
      // 背景扣除开关状态与修正后图
      const bgBox = document.getElementById('roi-bg');
      if (bgBox) bgBox.checked = !!(roiRes.roi && roiRes.roi.bg_subtract);
      const roiOv = document.getElementById('roi-overlay');
      if (roiOv) {
        if (roiRes.roi) {
          roiOv.src = '/api/images/' + currentImageId + '/overlay?t=' + Date.now();
          roiOv.closest('div').style.display = '';
        } else {
          roiOv.closest('div').style.display = 'none';
        }
      }
      // 编辑器底图用处理后图（选中图时已自动预处理）；无处理图则回退原图
      const imgUrl = '/api/images/' + currentImageId + '/processed?t=' + Date.now();
      const im = new Image();
      im.onload = function () {
        roiEditor.load(imgUrl, roiRes.roi || null);
      };
      im.onerror = function () {
        roiEditor.load('/api/images/' + currentImageId + '/original?t=' + Date.now(), roiRes.roi || null);
      };
      im.src = imgUrl;
      await loadTemplates();
      status.textContent = roiRes.roi
        ? ('当前 ROI：' + (roiRes.roi.source === 'manual' ? '手动' : '自动') + '（' + roiRes.roi.x + ',' + roiRes.roi.y + ',' + roiRes.roi.w + ',' + roiRes.roi.h + '）')
        : '尚未设置 ROI，可点击「自动套用模板」或直接在图上框选';
      status.className = 'status-line';
    } catch (e) {
      status.textContent = 'ROI 加载失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  async function runRoiAuto() {
    if (!currentImageId) return;
    const status = document.getElementById('roi-status');
    status.textContent = '自动套用模板...';
    try {
      const res = await global.API.post('/api/pipeline/' + currentImageId + '/roi/auto', {});
      ensureRoiEditor().setRoi(res.roi);
      status.textContent = '已自动识别检测区：' + JSON.stringify(res.roi);
      status.className = 'status-line';
      refreshRoiOverlay();
    } catch (e) {
      status.textContent = '自动套用失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  function refreshRoiOverlay() {
    const ov = document.getElementById('roi-overlay');
    if (!ov || !currentImageId) return;
    ov.src = '/api/images/' + currentImageId + '/overlay?t=' + Date.now();
    ov.closest('div').style.display = '';
  }

  async function saveRoiManual() {
    if (!currentImageId) return;
    const roi = ensureRoiEditor().getRoi();
    const status = document.getElementById('roi-status');
    if (!roi) {
      status.textContent = '请先在图上框选检测区';
      status.className = 'status-line err';
      return;
    }
    status.textContent = '保存中...';
    try {
      await global.API.post('/api/pipeline/' + currentImageId + '/roi', Object.assign({}, roi, {
        source: 'manual',
        bg_subtract: document.getElementById('roi-bg').checked ? 1 : 0,
      }));
      status.textContent = 'ROI 已保存，特征已级联重算';
      status.className = 'status-line';
      refreshRoiOverlay();
      showWfPanel(5);
    } catch (e) {
      status.textContent = '保存失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  async function saveRoiAsTemplate() {
    if (!currentImageId) return;
    const roi = ensureRoiEditor().getRoi();
    if (!roi) {
      document.getElementById('roi-status').textContent = '请先框选检测区';
      document.getElementById('roi-status').className = 'status-line err';
      return;
    }
    const name = prompt('模板名称：', '模板' + new Date().toLocaleDateString());
    if (!name) return;
    try {
      await global.API.post('/api/templates', Object.assign({}, roi, {
        name: name,
        ref_image_id: currentImageId,
      }));
      await loadTemplates();
      document.getElementById('roi-status').textContent = '已保存为模板：' + name;
      document.getElementById('roi-status').className = 'status-line';
    } catch (e) {
      document.getElementById('roi-status').textContent = '保存模板失败：' + e.message;
      document.getElementById('roi-status').className = 'status-line err';
    }
  }

  // ---- M3：特征 ----

  async function refreshFeatures() {
    if (!currentImageId) {
      document.getElementById('feat-status').textContent = '请先选中一张图';
      return;
    }
    const t = Date.now();
    // 处理后图（ROI 标注）、ROI 区域、ROI 平均色
    const ov = document.getElementById('feat-overlay');
    const cr = document.getElementById('feat-crop');
    const av = document.getElementById('feat-avg');
    try {
      const res = await global.API.get('/api/pipeline/' + currentImageId + '/features');
      const f = res.features;
      const cells = document.querySelectorAll('#feat-row td');
      if (!f) {
        cells.forEach(function (c) { c.textContent = '-'; });
        document.getElementById('feat-status').textContent = '尚无特征：请先完成 ROI 设置';
        ov.closest('div').style.display = 'none';
        return;
      }
      ov.src = '/api/images/' + currentImageId + '/overlay?t=' + t;
      cr.src = '/api/images/' + currentImageId + '/roi_crop?t=' + t;
      av.src = '/api/images/' + currentImageId + '/roi_avg?t=' + t;
      ov.closest('div').style.display = '';
      const vals = [f.mean_r, f.mean_g, f.mean_b, f.hue, f.saturation, f.value, f.ratio_gr, f.ratio_bg, f.intensity, f.texture_entropy];
      cells.forEach(function (c, i) { c.textContent = vals[i]; });
      // 通道分离 RGB 值（全图）与 ROI 平均 RGB 对照
      const chEl = document.getElementById('feat-ch');
      try {
        const ch = (await global.API.get('/api/pipeline/' + currentImageId + '/channels')).channels;
        if (ch && ch.mean_r != null) {
          chEl.innerHTML = '<table class="feat-table"><thead><tr><th></th><th>R</th><th>G</th><th>B</th></tr></thead><tbody>'
            + '<tr><td>通道分离（全图）</td><td>' + ch.mean_r + '</td><td>' + ch.mean_g + '</td><td>' + ch.mean_b + '</td></tr>'
            + '<tr><td>ROI 平均</td><td>' + f.mean_r + '</td><td>' + f.mean_g + '</td><td>' + f.mean_b + '</td></tr>'
            + '</tbody></table>'
            + '<div class="hint">ROI 平均取自检测区；全图通道均值供整体颜色观察参考。</div>';
        } else {
          chEl.textContent = '尚无通道分离数据：可在「通道分离」步骤点击「重新分离通道」。';
        }
      } catch (e2) {
        chEl.textContent = '通道数据读取失败：' + e2.message;
      }
      // 各 ROI 平均 RGB 柱状图（问题2-F 可视化）
      try {
        const all = await global.API.get('/api/images/' + currentImageId + '/features');
        const rois = all.rois || {};
        const names = Object.keys(rois);
        const barsEl = document.getElementById('feat-bars');
        if (names.length) {
          let html = '<table class="feat-table"><thead><tr><th>ROI</th><th>R</th><th>G</th><th>B</th></tr></thead><tbody>';
          names.forEach(function (n) {
            const rf = rois[n];
            const maxV = 255;
            html += '<tr><td><b>' + esc(n) + '</b></td>'
              + '<td><div class="bar"><div class="bar-fill" style="width:' + Math.min(100, rf.mean_r / maxV * 100) + '%;background:#e05656;"></div></div>' + Math.round(rf.mean_r) + '</td>'
              + '<td><div class="bar"><div class="bar-fill" style="width:' + Math.min(100, rf.mean_g / maxV * 100) + '%;background:#3fae6a;"></div></div>' + Math.round(rf.mean_g) + '</td>'
              + '<td><div class="bar"><div class="bar-fill" style="width:' + Math.min(100, rf.mean_b / maxV * 100) + '%;background:#4a7fd4;"></div></div>' + Math.round(rf.mean_b) + '</td></tr>';
          });
          html += '</tbody></table>';
          barsEl.innerHTML = html;
          barsEl.className = '';
        } else {
          barsEl.textContent = '尚无 ROI 特征：请先完成 ROI 设置并提取特征。';
        }
      } catch (e3) {
        document.getElementById('feat-bars').textContent = '柱状图加载失败：' + e3.message;
      }
      document.getElementById('feat-status').textContent = '特征已就绪（基于处理后图像）';
      document.getElementById('feat-status').className = 'status-line';
    } catch (e) {
      document.getElementById('feat-status').textContent = '特征读取失败：' + e.message;
      document.getElementById('feat-status').className = 'status-line err';
    }
  }

  async function runFeatures() {
    if (!currentImageId) return;
    const status = document.getElementById('feat-status');
    status.textContent = '提取中...';
    try {
      await global.API.post('/api/pipeline/' + currentImageId + '/features', {});
      await refreshFeatures();
    } catch (e) {
      status.textContent = '特征提取失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  async function runPreprocess(initial) {
    if (!currentImageId) return;
    const body = {
      filter: document.getElementById('pp-filter').value,
      kernel: parseInt(document.getElementById('pp-kernel').value, 10),
      use_dark: document.getElementById('pp-dark').checked,
      use_flat: document.getElementById('pp-flat').checked,
      use_wb: document.getElementById('pp-wb').checked,
      wb_roi_name: document.getElementById('pp-wb-roi').value.trim() || undefined,
    };
    const status = document.getElementById('pp-status');
    try {
      const res = await global.API.post('/api/pipeline/' + currentImageId + '/preprocess', body);
      const orig = document.getElementById('pp-img-orig');
      orig.src = '/api/images/' + currentImageId + '/original?t=' + Date.now();
      const proc = document.getElementById('pp-img-proc');
      proc.src = res.processed_url + '?t=' + Date.now();
      document.getElementById('pp-compare').style.display = 'block';
      // 对齐两张图：overlay 内图片宽度=容器宽度
      const box = document.getElementById('pp-compare-box');
      const boxW = box.clientWidth;
      proc.style.width = boxW + 'px';
      status.textContent = '预处理完成（' + res.params.filter + '，核 ' + res.params.kernel + '）';
      status.className = 'status-line';
      if (!initial && global.FluroApp) global.FluroApp.refreshTopbar();
    } catch (e) {
      status.textContent = '预处理失败：' + e.message;
      status.className = 'status-line err';
    }
  }

  function uploadRef(kind) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = async function () {
      if (!input.files || !input.files[0]) return;
      const fd = new FormData();
      fd.append('file', input.files[0]);
      const status = document.getElementById('pp-status');
      status.textContent = '上传参考图...';
      try {
        const res = await global.API.postForm('/api/refs/' + kind, fd);
        status.textContent = (kind === 'dark' ? '暗场' : '平场') + '参考图已保存：' + res.path;
        status.className = 'status-line';
        loadRefPreviews();
      } catch (e) {
        status.textContent = '参考图上传失败：' + e.message;
        status.className = 'status-line err';
      }
    };
    input.click();
  }

  // 参考图预览：加载并显示暗场/平场参考图缩略
  async function loadRefPreviews() {
    const map = { dark: 'pp-dark-prev', flat: 'pp-flat-prev' };
    try {
      const res = await global.API.get('/api/refs');
      Object.keys(map).forEach(function (kind) {
        const img = document.getElementById(map[kind]);
        const has = !!res[kind + '_path'];
        if (has) {
          img.src = '/api/refs/' + kind + '/image?t=' + Date.now();
          img.style.display = 'block';
        } else {
          img.style.display = 'none';
          img.removeAttribute('src');
        }
      });
    } catch (e) { /* 预览失败不阻塞 */ }
  }

  async function clearRef(kind) {
    try {
      await global.API.del('/api/refs/' + kind);
      document.getElementById('pp-status').textContent = (kind === 'dark' ? '暗场' : '平场') + '参考图已清除';
      document.getElementById('pp-status').className = 'status-line';
      loadRefPreviews();
    } catch (e) {
      document.getElementById('pp-status').textContent = '清除失败：' + e.message;
      document.getElementById('pp-status').className = 'status-line err';
    }
  }

  var initialized = false;

  function init() {
    if (initialized) {
      refreshGallery();
      return;
    }
    initialized = true;

    const kind = document.getElementById('up-kind');
    const concWrap = document.getElementById('up-conc-wrap');
    const btn = document.getElementById('up-btn');
    const status = document.getElementById('up-status');
    const filesInput = document.getElementById('up-files');

    // 自动处理全部：不强制人工，一键重跑流水线（自动完成预处理/ROI/特征）
    document.getElementById('up-auto').addEventListener('click', async function () {
      const list = await global.API.get('/api/images');
      const ids = list.images.map(function (i) { return i.id; });
      if (!ids.length) { status.textContent = '当前没有图片可处理'; return; }
      status.textContent = '自动处理全部（' + ids.length + ' 张）...';
      status.className = 'status-line';
      try {
        await runPipeline(ids);
      } catch (e) {
        status.textContent = '自动处理失败：' + e.message;
        status.className = 'status-line err';
      }
    });

    async function uploadFiles(fileList) {
      const files = fileList;
      if (!files || files.length === 0) {
        status.textContent = '请先选择图片文件';
        status.className = 'status-line err';
        return;
      }
      const fd = new FormData();
      for (const f of files) fd.append('files', f);
      fd.append('kind', kind.value);
      fd.append('batch', document.getElementById('up-batch').value || '');
      if (kind.value === 'calibration') {
        fd.append('known_conc', document.getElementById('up-conc').value || '');
        const unitSel = document.getElementById('up-conc-unit');
        fd.append('conc_unit', unitSel ? unitSel.value : 'ng/mL');
      }
      btn.disabled = true;
      status.textContent = '上传中（' + files.length + ' 个文件）...';
      status.className = 'status-line';
      try {
        const res = await global.API.postForm('/api/images/upload', fd);
        status.textContent = `上传完成：${res.images.length} 张` + (res.errors && res.errors.length ? `，失败 ${res.errors.length} 张（${res.errors.join('；')}）` : '');
        status.className = res.errors && res.errors.length ? 'status-line err' : 'status-line';
        renderGallery(res.images);
        filesInput.value = '';
        const dirInput = document.getElementById('up-dir');
        if (dirInput) dirInput.value = '';
        if (res.images.length) runPipeline(res.images.map(function (im) { return im.id; }));
        if (global.FluroApp) global.FluroApp.refreshTopbar();
      } catch (e) {
        status.textContent = '上传失败：' + e.message;
        status.className = 'status-line err';
      } finally {
        btn.disabled = false;
      }
    }

    btn.addEventListener('click', function () { uploadFiles(filesInput.files); });

    const dirInput = document.getElementById('up-dir');
    const dirBtn = document.getElementById('up-dir-btn');
    if (dirBtn) dirBtn.addEventListener('click', function () { dirInput.click(); });
    if (dirInput) dirInput.addEventListener('change', function () {
      // 只取目录中受支持的图片文件，避免把隐藏文件/临时文件传上去
      const IMG_EXT = /\.(jpe?g|png|bmp|tiff?|webp)$/i;
      const files = Array.prototype.filter.call(dirInput.files, function (f) {
        return IMG_EXT.test(f.name) && f.size > 0;
      });
      if (!files.length) {
        status.textContent = '所选文件夹中没有可导入的图片（jpg/jpeg/png/bmp/tif/tiff/webp）';
        status.className = 'status-line err';
        return;
      }
      uploadFiles(files);
    });

    document.getElementById('pp-run').addEventListener('click', function () { runPreprocess(false); });
    document.getElementById('ch-run').addEventListener('click', runChannels);
    document.getElementById('pp-slider').addEventListener('input', function (e) {
      document.getElementById('pp-overlay').style.width = e.target.value + '%';
    });
    document.getElementById('pp-ref-dark').addEventListener('click', function () { uploadRef('dark'); });
    document.getElementById('pp-ref-flat').addEventListener('click', function () { uploadRef('flat'); });
    document.getElementById('pp-dark-clear').addEventListener('click', function () { clearRef('dark'); });
    document.getElementById('pp-flat-clear').addEventListener('click', function () { clearRef('flat'); });
    loadRefPreviews();

    document.getElementById('roi-auto').addEventListener('click', runRoiAuto);
    document.getElementById('roi-save').addEventListener('click', saveRoiManual);
    document.getElementById('roi-save-tpl').addEventListener('click', saveRoiAsTemplate);
    document.getElementById('roi-tpl-select').addEventListener('change', async function (e) {
      const tid = e.target.value;
      if (!tid) return;
      try { await global.API.post('/api/templates/' + tid + '/activate', {}); await loadTemplates(tid); } catch (err) { /* 忽略 */ }
    });
    document.getElementById('feat-run').addEventListener('click', runFeatures);

    // 步骤条点击：步骤 1 回上传；步骤 2/3/4 需先选中图
    document.querySelectorAll('.wf-step').forEach(function (s) {
      s.addEventListener('click', function () {
        const n = parseInt(s.dataset.wf, 10);
        if (n === 1) { showWfPanel(1); return; }
        if (!currentImageId) {
          document.getElementById('roi-status').textContent = '请先在步骤 ① 上传并点击选中一张图片';
          return;
        }
        showWfPanel(n);
      });
    });
  }

  global.FluroWorkflow = {
    init: init,
    renderGallery: renderGallery,
    selectImage: selectImage,
    openStep: openStep,
    showWfPanel: showWfPanel,
    onView: function (name) {
      if (name === 'workflow') init();
    },
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
