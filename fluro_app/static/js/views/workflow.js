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
      info.textContent = im.known_conc != null ? ('浓度 ' + im.known_conc) : (im.batch ? ('批次 ' + im.batch) : '');
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

  async function runPipeline(ids) {
    const el = document.getElementById('pipe-status');
    el.innerHTML = '<span class="pb pb-uploaded">自动流水线执行中...</span>';
    try {
      const res = await global.API.post('/api/pipeline/run', { image_ids: ids });
      renderPipeStatus(res.results);
      await refreshGallery();
      if (global.FluroApp) global.FluroApp.refreshTopbar();
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
        zone.innerHTML = '<div class="card">'
          + '<h3>标定数据录入（写浓度 → 建模型）</h3>'
          + '<p class="hint">把这张标定图写入已知浓度并加入标定数据集；再到「标定建模」页拟合并保存生效模型。</p>'
          + '<div class="form-row">'
          + '<label>浓度（ng/mL）：<input id="cal-conc" type="number" step="any" value="' + (im.known_conc != null ? im.known_conc : '') + '"></label>'
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
      msg.textContent = '已加入浓度 ' + res.conc + (res.reused ? '（更新已有数据点）' : '') + '，请到「标定建模」页拟合并保存模型';
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
    zone.innerHTML =
      '<div class="det-card"><div class="det-main">'
      + '<div class="det-conc">' + Number(d.conc).toFixed(2) + ' <span class="det-unit">ng/mL</span></div>'
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

    btn.addEventListener('click', async function () {
      const files = filesInput.files;
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
      }
      btn.disabled = true;
      status.textContent = '上传中...';
      status.className = 'status-line';
      try {
        const res = await global.API.postForm('/api/images/upload', fd);
        status.textContent = `上传完成：${res.images.length} 张` + (res.errors && res.errors.length ? `，失败 ${res.errors.length} 张（${res.errors.join('；')}）` : '');
        status.className = res.errors && res.errors.length ? 'status-line err' : 'status-line';
        renderGallery(res.images);
        filesInput.value = '';
        if (res.images.length) runPipeline(res.images.map(function (im) { return im.id; }));
        if (global.FluroApp) global.FluroApp.refreshTopbar();
      } catch (e) {
        status.textContent = '上传失败：' + e.message;
        status.className = 'status-line err';
      } finally {
        btn.disabled = false;
      }
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
