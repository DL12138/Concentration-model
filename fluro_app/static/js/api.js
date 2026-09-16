/* api.js：fetch 封装，全部走本地服务 */
(function (global) {
  'use strict';

  async function handle(resp) {
    const ct = resp.headers.get('content-type') || '';
    const body = ct.includes('application/json') ? await resp.json() : await resp.text();
    if (!resp.ok) {
      const msg = (body && body.error) ? body.error : ('HTTP ' + resp.status);
      throw new Error(msg);
    }
    return body;
  }

  global.API = {
    async get(url) {
      const resp = await fetch(url);
      return handle(resp);
    },
    async post(url, data) {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data || {}),
      });
      return handle(resp);
    },
    async postForm(url, formData) {
      const resp = await fetch(url, { method: 'POST', body: formData });
      return handle(resp);
    },
    async put(url, data) {
      const resp = await fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data || {}),
      });
      return handle(resp);
    },
    async del(url) {
      const resp = await fetch(url, { method: 'DELETE' });
      return handle(resp);
    },
    async postDownload(url, data, filename) {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data || {}),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error((body && body.error) || ('HTTP ' + resp.status));
      }
      const blob = await resp.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename || 'download';
      a.click();
      URL.revokeObjectURL(a.href);
      return true;
    },
  };
})(window);
