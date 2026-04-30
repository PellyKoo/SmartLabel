/**
 * SmartLabel 前端主逻辑
 *
 * 职责：
 * - Tab 路由切换
 * - 引擎面板：加载 / 释放 / 状态轮询
 * - 各页面：提交任务 → 轮询进度 → 展示结果
 */

// ==================== 工具 ====================

function qs(sel, ctx = document) { return ctx.querySelector(sel); }
function qsa(sel, ctx = document) { return [...ctx.querySelectorAll(sel)]; }

function log(msg, level = 'info') {
  const el = qs('#log-area');
  if (!el) return;
  const colors = { error: '#EF5350', warn: '#FF9800', info: '#D4D4D4' };
  const time = new Date().toTimeString().slice(0, 8);
  el.innerHTML += `<span style="color:#666">[${time}]</span> <span style="color:${colors[level] || colors.info}">${msg}</span>\n`;
  el.scrollTop = el.scrollHeight;
}

function setStatus(msg) {
  const el = qs('#status-bar');
  if (el) el.textContent = msg;
}

function badge(text, type = 'dim') {
  return `<span class="badge badge-${type}">${text}</span>`;
}

function setProgress(wrapId, pct) {
  const el = qs(`#${wrapId} .progress-bar`);
  if (el) el.style.width = Math.min(100, pct) + '%';
}

let _pollers = {};

function startPolling(taskId, onUpdate, intervalMs = 1000) {
  stopPolling(taskId);
  _pollers[taskId] = setInterval(async () => {
    try {
      const s = await API.taskStatus(taskId);
      onUpdate(s);
      if (['completed', 'failed', 'cancelled'].includes(s.status)) {
        stopPolling(taskId);
      }
    } catch (e) {
      log(`轮询失败: ${e.message}`, 'error');
      stopPolling(taskId);
    }
  }, intervalMs);
}

function stopPolling(taskId) {
  if (_pollers[taskId]) {
    clearInterval(_pollers[taskId]);
    delete _pollers[taskId];
  }
}

// ==================== Tab 路由 ====================

function switchTab(name) {
  qsa('.nav-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  qsa('.tab-page').forEach(p => p.classList.toggle('active', p.id === `page-${name}`));
}

// ==================== 引擎面板 ====================

let _engineKey = '';
let _engineCfg = {};

/** 根据类型 + 路径生成稳定的引擎 Key（相同配置 = 相同 key，pool 不重复加载）*/
function _autoEngineKey(type, path) {
  let h = 5381;
  const s = type + '|' + path;
  for (let i = 0; i < s.length; i++) h = (h * 33 ^ s.charCodeAt(i)) >>> 0;
  return `${type}_${h.toString(36).slice(0, 6)}`;
}

function _setLoadStatus(text, color = '#888') {
  const el = qs('#engine-load-status');
  if (el) { el.textContent = text; el.style.color = color; }
}

async function refreshEngineStatus() {
  try {
    const data = await API.engineStatus();
    const loaded = data.loaded_engines || [];
    const panel = qs('#engine-status-list');
    if (!panel) return;
    if (loaded.length === 0) {
      panel.innerHTML = '<span class="text-dim text-sm">无已加载引擎</span>';
    } else {
      panel.innerHTML = loaded.map(e => `
        <div class="flex-row mt8">
          <span class="badge badge-ok">●</span>
          <span class="text-sm" style="flex:1;overflow:hidden;text-overflow:ellipsis"
                title="${e.engine_key}">${e.engine_key}</span>
          <button class="danger" onclick="unloadEngine('${e.engine_key}')"
                  style="padding:2px 8px;font-size:11px;flex-shrink:0">释放</button>
        </div>
        <div class="text-dim text-sm" style="margin-left:12px">
          ${e.gpu_memory_allocated_gb !== undefined
            ? `显存: ${e.gpu_memory_allocated_gb}GB / ${e.gpu_memory_total_gb}GB` : ''}
        </div>`).join('');
    }
  } catch (e) {
    log(`刷新引擎状态失败: ${e.message}`, 'warn');
  }
}

async function loadEngine() {
  const type  = qs('#engine-type').value;
  const path  = qs('#model-path').value.trim();
  const quant = qs('#quantization').value;

  if (!path) {
    _setLoadStatus('请先填写模型路径', '#FF9800');
    log('模型路径不能为空', 'warn');
    return;
  }

  // 自动生成 key，相同配置 key 相同，引擎池不重复加载
  const key = _autoEngineKey(type, path);
  const cfg = {
    type,
    vlm: { model_path: path, quantization: quant, device: 'cuda:0',
           torch_dtype: 'float16', max_new_tokens: 256, multi_sample: false },
  };
  _engineKey = key;
  _engineCfg = cfg;

  // 显示 key（只读）
  const keyDisplay = qs('#engine-key-display');
  const keyRow     = qs('#engine-key-row');
  if (keyDisplay) keyDisplay.textContent = key;
  if (keyRow)     keyRow.style.display = '';

  // 按钮禁用 + 三处即时反馈：状态文本 + 状态栏 + 日志区
  const btn = qs('#engine-load-btn');
  if (btn) {
    btn.disabled = true;
    btn.textContent = '加载中...';
  }
  _setLoadStatus('加载中...', '#FF9800');
  setStatus(`加载引擎 ${key}（VLM 模型可能需要 30-60 秒）...`);
  log(`▶ 提交加载请求: key=${key}, type=${type}, path=${path}`, 'info');

  try {
    // 后端立即返回 loading，加载在服务端后台线程进行
    await API.loadEngine(key, cfg);
    log(`✓ 加载请求已提交，开始轮询状态（每 2 秒）`, 'info');
    _pollEngineLoad(key, btn);
  } catch (e) {
    _setLoadStatus('请求失败 ✗', '#EF5350');
    log(`✗ 加载请求失败: ${e.message}`, 'error');
    if (btn) { btn.disabled = false; btn.textContent = '加载引擎'; }
  }
}

function _pollEngineLoad(key, btn) {
  let dots = 0;
  const t0 = Date.now();
  const restoreBtn = () => {
    if (btn) { btn.disabled = false; btn.textContent = '加载引擎'; }
  };
  const timer = setInterval(async () => {
    try {
      const s = await API.engineLoadStatus(key);
      dots = (dots + 1) % 4;
      const elapsed = Math.floor((Date.now() - t0) / 1000);
      if (s.status === 'loading' || s.status === 'not_started') {
        _setLoadStatus(`加载中${'.'.repeat(dots + 1)} (${elapsed}s)`, '#FF9800');
      } else if (s.status === 'ready') {
        clearInterval(timer);
        _setLoadStatus(`加载成功 ✓ (耗时 ${elapsed}s)`, '#4CAF50');
        log(`✓ 引擎已加载: ${key}（耗时 ${elapsed}s）`, 'info');
        setStatus('引擎已就绪');
        restoreBtn();
        await refreshEngineStatus();
      } else if (s.status === 'failed') {
        clearInterval(timer);
        _setLoadStatus('加载失败 ✗', '#EF5350');
        log(`✗ 引擎加载失败: ${s.error}`, 'error');
        setStatus('引擎加载失败');
        restoreBtn();
      }
    } catch (e) {
      clearInterval(timer);
      _setLoadStatus('轮询失败', '#EF5350');
      log(`✗ 状态轮询失败: ${e.message}`, 'error');
      restoreBtn();
    }
  }, 2000);
}

async function unloadEngine(key) {
  try {
    await API.unloadEngine(key);
    log(`引擎已释放: ${key}`);
    await refreshEngineStatus();
  } catch (e) {
    log(`释放失败: ${e.message}`, 'error');
  }
}

/**
 * 任务提交前确认引擎就绪。
 * 没加载 → 提示用户先加载；加载中 → 提示等待。
 */
async function _ensureEngineReady() {
  if (!_engineKey) {
    log('请先在左侧引擎面板加载引擎', 'warn');
    return false;
  }
  try {
    const s = await API.engineLoadStatus(_engineKey);
    if (s.status === 'ready') return true;
    if (s.status === 'loading' || s.status === 'not_started') {
      log(`引擎正在加载中，请等待加载完成后再提交任务`, 'warn');
      return false;
    }
    if (s.status === 'failed') {
      log(`引擎加载失败，请重新加载: ${s.error || ''}`, 'error');
      return false;
    }
    log(`引擎状态未知: ${s.status}`, 'warn');
    return false;
  } catch (e) {
    log(`检查引擎状态失败: ${e.message}`, 'error');
    return false;
  }
}

// ==================== 预标注页面 ====================

let _paTaskId = '';

async function startPreannotate() {
  const imageDir  = qs('#pa-image-dir').value.trim();
  const outputDir = qs('#pa-output-dir').value.trim();
  const cats      = qs('#pa-categories').value.trim().split(',').map(s => s.trim()).filter(Boolean);
  const task      = qs('#pa-task').value;

  if (!await _ensureEngineReady()) return;
  if (!imageDir || !outputDir) { log('图片目录和输出目录不能为空', 'warn'); return; }

  const btn = qs('#pa-start-btn');
  btn.disabled = true;

  try {
    let resp;
    if (task === 'classification') {
      if (!cats.length) { log('请填写分类类别', 'warn'); btn.disabled = false; return; }
      resp = await API.startPreannotateCls({
        engine_key: _engineKey, engine_config: _engineCfg,
        image_dir: imageDir, output_dir: outputDir, categories: cats,
      });
    } else {
      if (!cats.length) { log('请填写检测目标', 'warn'); btn.disabled = false; return; }
      resp = await API.startPreannotateDet({
        engine_key: _engineKey, engine_config: _engineCfg,
        image_dir: imageDir, output_dir: outputDir, targets: cats,
      });
    }
    _paTaskId = resp.task_id;
    log(`预标注任务已提交: ${_paTaskId}`);
    qs('#pa-stop-btn').disabled = false;
    startPolling(_paTaskId, (s) => updatePreannotateStatus(s));
  } catch (e) {
    log(`提交失败: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

function updatePreannotateStatus(s) {
  const [cur, tot] = s.progress || [0, 0];
  const pct = tot > 0 ? (cur / tot * 100) : 0;
  setProgress('pa-progress', pct);
  qs('#pa-status').innerHTML = `${badge(s.status, statusBadge(s.status))} ${cur}/${tot}`;
  if (s.status === 'completed') {
    log(`预标注完成: ${_paTaskId}`);
    setStatus('预标注完成');
    qs('#pa-start-btn').disabled = false;
    qs('#pa-stop-btn').disabled = true;
    showPreannotateResult();
  } else if (s.status === 'failed') {
    log(`预标注失败: ${s.error}`, 'error');
    qs('#pa-start-btn').disabled = false;
  }
}

async function showPreannotateResult() {
  try {
    const r = await API.taskResult(_paTaskId);
    const results = r.results || [];
    const table = qs('#pa-result-table');
    if (!table) return;
    table.innerHTML = `<tr><th>文件</th><th>预测类别</th><th>置信度</th><th>状态</th></tr>`
      + results.slice(0, 200).map(row => `
      <tr>
        <td>${(row.image_path || '').split(/[\\/]/).pop()}</td>
        <td>${row.predicted_class || ''}</td>
        <td>${row.confidence != null ? row.confidence.toFixed(2) : '-'}</td>
        <td>${row.is_uncertain ? badge('不确定', 'warn') : badge('OK', 'ok')}</td>
      </tr>`).join('');
  } catch (e) {
    log(`获取结果失败: ${e.message}`, 'error');
  }
}

async function stopPreannotate() {
  if (_paTaskId) await API.stopTask(_paTaskId);
  qs('#pa-stop-btn').disabled = true;
}

// ==================== 质检页面 ====================

let _qcTaskId = '';

async function startQC() {
  const imageDir  = qs('#qc-image-dir').value.trim();
  const annDir    = qs('#qc-annotation-dir').value.trim();
  const outputDir = qs('#qc-output-dir').value.trim();
  const cats      = qs('#qc-categories').value.trim().split(',').map(s => s.trim()).filter(Boolean);
  if (!await _ensureEngineReady()) return;
  if (!imageDir || !annDir || !outputDir || !cats.length) {
    log('请填写所有必填项', 'warn'); return;
  }
  const btn = qs('#qc-start-btn');
  btn.disabled = true;
  try {
    const resp = await API.startQCCls({
      engine_key: _engineKey, engine_config: _engineCfg,
      image_dir: imageDir, annotation_dir: annDir,
      output_dir: outputDir, categories: cats,
    });
    _qcTaskId = resp.task_id;
    log(`质检任务已提交: ${_qcTaskId}`);
    qs('#qc-stop-btn').disabled = false;
    startPolling(_qcTaskId, updateQCStatus);
  } catch (e) {
    log(`提交失败: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

function updateQCStatus(s) {
  const [cur, tot] = s.progress || [0, 0];
  const pct = tot > 0 ? (cur / tot * 100) : 0;
  setProgress('qc-progress', pct);
  qs('#qc-status').innerHTML = `${badge(s.status, statusBadge(s.status))} ${cur}/${tot}`;
  if (s.status === 'completed') {
    log(`质检完成: ${_qcTaskId}`);
    qs('#qc-start-btn').disabled = false;
    qs('#qc-stop-btn').disabled = true;
    showQCResult();
  } else if (s.status === 'failed') {
    log(`质检失败: ${s.error}`, 'error');
    qs('#qc-start-btn').disabled = false;
  }
}

async function showQCResult() {
  try {
    const r = await API.taskResult(_qcTaskId);
    qs('#qc-summary').innerHTML = `
      总检查: <b>${r.total_checked}</b> &nbsp;
      通过: <b class="text-green">${r.pass_count}</b> &nbsp;
      异议率: <b class="text-red">${(r.review_ratio * 100).toFixed(1)}%</b> &nbsp;
      VLM升级: <b>${r.escalated_count}</b>`;
    const samples = await API.reviewSamples(_qcTaskId);
    const table = qs('#qc-review-table');
    table.innerHTML = `<tr><th>文件</th><th>人工</th><th>引擎</th><th>置信度</th><th>VLM理由</th></tr>`
      + (samples || []).slice(0, 100).map(s => `
      <tr>
        <td><img src="${API.imageUrl(s.image_path)}" class="img-thumb" onerror="this.style.display='none'"
             onclick="previewImg('${s.image_path}')" title="${s.image_path.split(/[\\/]/).pop()}"></td>
        <td>${s.human_label}</td>
        <td class="text-red">${s.engine_label}</td>
        <td>${s.confidence != null ? s.confidence.toFixed(2) : '-'}</td>
        <td class="text-dim text-sm">${s.vlm_reason || ''}</td>
      </tr>`).join('');
  } catch (e) {
    log(`获取质检结果失败: ${e.message}`, 'error');
  }
}

async function stopQC() {
  if (_qcTaskId) await API.stopTask(_qcTaskId);
  qs('#qc-stop-btn').disabled = true;
}

// ==================== 视频页面 ====================

let _videoTaskId = '';

async function startVideo() {
  const videoDir  = qs('#video-dir').value.trim();
  const outputDir = qs('#video-output-dir').value.trim();
  const cats      = qs('#video-categories').value.trim().split(',').map(s => s.trim()).filter(Boolean);
  const strategy  = qs('#video-strategy').value;
  if (!await _ensureEngineReady()) return;
  if (!videoDir || !outputDir || !cats.length) { log('请填写所有必填项', 'warn'); return; }
  const btn = qs('#video-start-btn');
  btn.disabled = true;
  try {
    const resp = await API.startVideoClassify({
      engine_key: _engineKey, engine_config: _engineCfg,
      video_dir: videoDir, output_dir: outputDir,
      categories: cats, strategy,
    });
    _videoTaskId = resp.task_id;
    log(`视频分类任务已提交: ${_videoTaskId}`);
    qs('#video-stop-btn').disabled = false;
    startPolling(_videoTaskId, updateVideoStatus);
  } catch (e) {
    log(`提交失败: ${e.message}`, 'error');
    btn.disabled = false;
  }
}

function updateVideoStatus(s) {
  const [cur, tot] = s.progress || [0, 0];
  const pct = tot > 0 ? (cur / tot * 100) : 0;
  setProgress('video-progress', pct);
  qs('#video-status').innerHTML = `${badge(s.status, statusBadge(s.status))} ${cur}/${tot}`;
  if (s.status === 'completed') {
    log(`视频分类完成: ${_videoTaskId}`);
    qs('#video-start-btn').disabled = false;
    qs('#video-stop-btn').disabled = true;
    showVideoResult();
  } else if (s.status === 'failed') {
    log(`视频分类失败: ${s.error}`, 'error');
    qs('#video-start-btn').disabled = false;
  }
}

async function showVideoResult() {
  try {
    const r = await API.timeline(_videoTaskId);
    const results = Array.isArray(r) ? r : [r];
    const container = qs('#video-result');
    container.innerHTML = '';
    for (const vr of results) {
      const clips = vr.clips || [];
      const stats = vr.statistics || {};
      const total = Object.values(stats).reduce((a, b) => a + b, 0) || 1;

      // SVG 时间轴
      const COLORS = ['#4CAF50','#FF9800','#EF5350','#2196F3','#9C27B0','#00BCD4'];
      const labels = [...new Set(clips.map(c => c.label))];
      const colorMap = Object.fromEntries(labels.map((l, i) => [l, COLORS[i % COLORS.length]]));
      const svgW = 800, svgH = 40;
      const rects = clips.map(c => {
        const x = (c.start_sec / total) * svgW;
        const w = Math.max(1, ((c.end_sec - c.start_sec) / total) * svgW);
        return `<rect x="${x.toFixed(1)}" y="0" width="${w.toFixed(1)}" height="${svgH}"
                  fill="${colorMap[c.label]}" opacity=".85">
                  <title>${c.label} ${c.start_sec.toFixed(1)}s-${c.end_sec.toFixed(1)}s</title></rect>`;
      }).join('');
      const legend = labels.map(l =>
        `<span style="color:${colorMap[l]}">■</span> ${l}: ${(stats[l]||0).toFixed(1)}s`
      ).join(' &nbsp; ');

      const tableRows = clips.map(c => `
        <tr><td>${c.start_sec.toFixed(1)}</td><td>${c.end_sec.toFixed(1)}</td>
        <td>${c.label}</td><td>${(c.duration_sec||0).toFixed(1)}</td></tr>`).join('');

      container.innerHTML += `
        <div class="card">
          <h3>${(vr.video_path||'').split(/[\\/]/).pop()}</h3>
          <div id="timeline-container">
            <svg id="timeline-svg" viewBox="0 0 ${svgW} ${svgH}" preserveAspectRatio="none"
                 style="height:${svgH}px">${rects}</svg>
          </div>
          <div class="text-sm mt8">${legend}</div>
          <table class="mt8"><tr><th>开始(s)</th><th>结束(s)</th><th>标签</th><th>时长(s)</th></tr>
          ${tableRows}</table>
        </div>`;
    }
  } catch (e) {
    log(`获取视频结果失败: ${e.message}`, 'error');
  }
}

async function stopVideo() {
  if (_videoTaskId) await API.stopTask(_videoTaskId);
  qs('#video-stop-btn').disabled = true;
}

// ==================== 图片预览弹窗 ====================

function previewImg(path) {
  let modal = qs('#img-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'img-modal';
    modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.8);'
      + 'display:flex;align-items:center;justify-content:center;z-index:9999;cursor:pointer;';
    modal.onclick = () => modal.remove();
    document.body.appendChild(modal);
  }
  modal.innerHTML = `<img src="${API.imageUrl(path)}" style="max-width:90%;max-height:90%;border-radius:4px">`;
}

// ==================== 辅助 ====================

function statusBadge(s) {
  return { completed: 'ok', failed: 'err', running: 'running', queued: 'dim', cancelled: 'warn' }[s] || 'dim';
}

// ==================== 初始化 ====================

document.addEventListener('DOMContentLoaded', () => {
  // Tab 切换
  qsa('.nav-tab').forEach(t => t.addEventListener('click', () => switchTab(t.dataset.tab)));

  // 定时刷新引擎状态
  refreshEngineStatus();
  setInterval(refreshEngineStatus, 10000);

  // 任务历史页
  const histTab = qs('[data-tab="history"]');
  if (histTab) histTab.addEventListener('click', loadHistory);

  setStatus('SmartLabel Web 已就绪');
  log('SmartLabel Web 前端已加载');
});

async function loadHistory() {
  try {
    const tasks = await API.listTasks();
    const tbody = qs('#history-table tbody');
    if (!tbody) return;
    tbody.innerHTML = tasks.slice().reverse().map(t => `
      <tr>
        <td class="text-sm">${t.id}</td>
        <td>${t.type}</td>
        <td>${badge(t.status, statusBadge(t.status))}</td>
        <td>${(t.progress||[0,0])[0]}/${(t.progress||[0,0])[1]}</td>
        <td class="text-dim text-sm">${t.created_at.slice(11,19)}</td>
        <td><button onclick="stopPolling('${t.id}');API.stopTask('${t.id}')" ${['completed','failed','cancelled'].includes(t.status)?'disabled':''}>停止</button></td>
      </tr>`).join('');
  } catch (e) {
    log(`加载历史失败: ${e.message}`, 'error');
  }
}
