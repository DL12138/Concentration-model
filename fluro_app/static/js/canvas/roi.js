/* canvas/roi.js：ROI 框选编辑器（Canvas）
   支持：框选创建、整体拖动、四角缩放、归一化坐标输出 */
(function (global) {
  'use strict';

  function create(canvas, opts) {
    opts = opts || {};
    var ctx = canvas.getContext('2d');
    var img = null;         // HTMLImageElement
    var roi = null;         // 归一化 {x,y,w,h}
    var imgSize = { w: 0, h: 0 };
    var scale = 1, ox = 0, oy = 0;   // 图片在 canvas 中的布局
    var dragging = null;    // 'create'|'move'|'nw'|'ne'|'sw'|'se'
    var start = null;       // 拖动起点（图片像素坐标）
    var startRoi = null;
    var HANDLE = 10;        // 手柄命中半径（canvas px）

    function layout() {
      var cw = canvas.width, ch = canvas.height;
      scale = Math.min(cw / imgSize.w, ch / imgSize.h);
      ox = (cw - imgSize.w * scale) / 2;
      oy = (ch - imgSize.h * scale) / 2;
    }

    function toImg(e) {
      var r = canvas.getBoundingClientRect();
      var px = (e.clientX - r.left - ox) / scale;
      var py = (e.clientY - r.top - oy) / scale;
      return { x: px, y: py };
    }

    function inImg(p) {
      return p.x >= 0 && p.y >= 0 && p.x <= imgSize.w && p.y <= imgSize.h;
    }

    function clampRoi(r) {
      var x = Math.max(0, Math.min(r.x, imgSize.w - 1));
      var y = Math.max(0, Math.min(r.y, imgSize.h - 1));
      var w = Math.max(1, Math.min(r.w, imgSize.w - x));
      var h = Math.max(1, Math.min(r.h, imgSize.h - y));
      return { x: x, y: y, w: w, h: h };
    }

    function toNorm(r) {
      if (!imgSize.w || !imgSize.h) return null;
      return {
        x: +(r.x / imgSize.w).toFixed(4),
        y: +(r.y / imgSize.h).toFixed(4),
        w: +(r.w / imgSize.w).toFixed(4),
        h: +(r.h / imgSize.h).toFixed(4),
      };
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!img) return;
      ctx.drawImage(img, ox, oy, imgSize.w * scale, imgSize.h * scale);
      if (!roi) return;
      var x = ox + roi.x * scale, y = oy + roi.y * scale;
      var w = roi.w * scale, h = roi.h * scale;
      ctx.fillStyle = 'rgba(15,157,143,0.18)';
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = '#0f9d8f';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = '#0f9d8f';
      var hs = HANDLE / 2;
      [[x, y], [x + w, y], [x, y + h], [x + w, y + h]].forEach(function (p) {
        ctx.fillRect(p[0] - hs, p[1] - hs, HANDLE, HANDLE);
      });
    }

    function hitHandle(p) {
      if (!roi) return null;
      var x = ox + roi.x * scale, y = oy + roi.y * scale;
      var w = roi.w * scale, h = roi.h * scale;
      var corners = { nw: [x, y], ne: [x + w, y], sw: [x, y + h], se: [x + w, y + h] };
      var e = canvas.getBoundingClientRect();
      var cx = e.clientX, cy = e.clientY;
      for (var k in corners) {
        var c = corners[k];
        if (Math.abs(cx - c.x - e.left) <= HANDLE && Math.abs(cy - c.y - e.top) <= HANDLE) return k;
      }
      return null;
    }

    function onDown(e) {
      var p = toImg(e);
      if (!inImg(p)) return;
      var h = hitHandle(e);
      if (h) { dragging = h; start = p; startRoi = { x: roi.x, y: roi.y, w: roi.w, h: roi.h }; return; }
      if (roi && p.x >= roi.x && p.y >= roi.y && p.x <= roi.x + roi.w && p.y <= roi.y + roi.h) {
        dragging = 'move'; start = p; startRoi = { x: roi.x, y: roi.y, w: roi.w, h: roi.h }; return;
      }
      // 无 ROI 或点在框外：开始新建
      dragging = 'create';
      start = p;
      roi = { x: p.x, y: p.y, w: 1, h: 1 };
    }

    function onMove(e) {
      if (!dragging) return;
      var p = toImg(e);
      var r = { x: roi.x, y: roi.y, w: roi.w, h: roi.h };
      if (dragging === 'create' || dragging === 'nw') {
        r.x = Math.min(start.x, p.x); r.y = Math.min(start.y, p.y);
        r.w = Math.abs(p.x - start.x); r.h = Math.abs(p.y - start.y);
      } else if (dragging === 'ne') {
        r.y = Math.min(start.y, p.y); r.h = Math.abs(p.y - start.y);
        r.w = Math.abs(p.x - startRoi.x);
      } else if (dragging === 'sw') {
        r.x = Math.min(start.x, p.x); r.w = Math.abs(p.x - start.x);
        r.h = Math.abs(p.y - startRoi.y);
      } else if (dragging === 'se') {
        r.w = Math.abs(p.x - startRoi.x); r.h = Math.abs(p.y - startRoi.y);
      } else if (dragging === 'move') {
        r.x = startRoi.x + (p.x - start.x);
        r.y = startRoi.y + (p.y - start.y);
      }
      roi = clampRoi(r);
      draw();
      if (opts.onRoiChange) opts.onRoiChange(toNorm(roi));
    }

    function onUp() {
      if (dragging === 'create' && roi && (roi.w < 3 || roi.h < 3)) roi = null;
      dragging = null;
      draw();
      if (opts.onRoiChange) opts.onRoiChange(toNorm(roi));
    }

    canvas.addEventListener('mousedown', onDown);
    canvas.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);

    return {
      load: function (url, r) {
        var im = new Image();
        im.onload = function () {
          img = im;
          imgSize = { w: im.naturalWidth, h: im.naturalHeight };
          roi = r ? { x: r.x * im.naturalWidth, y: r.y * im.naturalHeight, w: r.w * im.naturalWidth, h: r.h * im.naturalHeight } : null;
          layout();
          draw();
        };
        im.src = url;
      },
      getRoi: function () { return toNorm(roi); },
      setRoi: function (norm) {
        if (!norm) { roi = null; draw(); return; }
        roi = { x: norm.x * imgSize.w, y: norm.y * imgSize.h, w: norm.w * imgSize.w, h: norm.h * imgSize.h };
        draw();
      },
      clear: function () { roi = null; draw(); },
    };
  }

  global.ROIEditor = { create: create };
})(window);
