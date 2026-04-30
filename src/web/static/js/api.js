/**
 * SmartLabel API 封装（fetch wrapper）
 */
const API = {
  async _req(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const ct = resp.headers.get('content-type') || '';
    return ct.includes('application/json') ? resp.json() : resp.text();
  },

  // ---- 引擎 ----
  loadEngine:      (key, cfg) => API._req('POST', '/api/engine/load', { engine_key: key, engine_config: cfg }),
  engineLoadStatus:(key)      => API._req('GET',  `/api/engine/load-status/${key}`),
  unloadEngine:    (key)      => API._req('POST', `/api/engine/unload/${key}`, undefined),
  engineStatus:    ()         => API._req('GET',  '/api/engine/status'),
  engineStatusOne: (key)      => API._req('GET',  `/api/engine/status/${key}`),

  // ---- 任务 ----
  listTasks:    ()         => API._req('GET',  '/api/tasks'),
  taskStatus:   (id)       => API._req('GET',  `/api/tasks/${id}/status`),
  taskResult:   (id)       => API._req('GET',  `/api/tasks/${id}/result`),
  stopTask:     (id)       => API._req('POST', `/api/tasks/${id}/stop`),
  reviewSamples:(id)       => API._req('GET',  `/api/tasks/${id}/review-samples`),
  timeline:     (id)       => API._req('GET',  `/api/tasks/${id}/timeline`),

  // ---- 预标注 ----
  startPreannotateCls: (data) => API._req('POST', '/api/preannotate/classification/start', data),
  startPreannotateDet: (data) => API._req('POST', '/api/preannotate/detection/start',      data),

  // ---- 质检 ----
  startQCCls: (data) => API._req('POST', '/api/qualitycheck/classification/start', data),
  startQCDet: (data) => API._req('POST', '/api/qualitycheck/detection/start',      data),

  // ---- 视频 ----
  startVideoClassify: (data) => API._req('POST', '/api/video/classify/start', data),

  // ---- 图片代理 ----
  imageUrl: (path) => `/api/image?path=${encodeURIComponent(path)}`,

  // ---- 文件系统 ----
  fsLs:   (path) => API._req('GET', `/api/fs/ls?path=${encodeURIComponent(path || '')}`),
  fsRead: (path) => API._req('GET', `/api/fs/read?path=${encodeURIComponent(path)}`),
};
