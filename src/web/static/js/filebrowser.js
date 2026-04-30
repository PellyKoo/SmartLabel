/**
 * 服务器端文件 / 目录浏览器
 *
 * 用法：
 *   FileBrowser.pickDir(callback)          弹窗让用户选一个目录，回调 (path)
 *   FileBrowser.pickFile(exts, callback)   弹窗让用户选一个文件，回调 (path)
 *   FileBrowser.attachTo(inputEl, mode)    给 input 挂一个"浏览"按钮（mode: 'dir'|'file'）
 */
const FileBrowser = (() => {
  let _resolve = null;
  let _mode = 'dir';      // 'dir' | 'file'
  let _currentPath = '';
  let _modal = null;
  let _listEl = null;
  let _pathEl = null;
  let _confirmBtn = null;

  // ---- 创建 Modal DOM（仅一次）----
  function _ensureModal() {
    if (_modal) return;

    _modal = document.createElement('div');
    _modal.id = 'fb-modal';
    _modal.innerHTML = `
<div id="fb-backdrop"></div>
<div id="fb-dialog">
  <div id="fb-header">
    <span id="fb-title">选择目录</span>
    <button id="fb-close" title="关闭">✕</button>
  </div>
  <div id="fb-toolbar">
    <button id="fb-up" title="上一级">↑ 上级</button>
    <span id="fb-path-display"></span>
  </div>
  <div id="fb-list"></div>
  <div id="fb-footer">
    <input id="fb-manual" type="text" placeholder="或直接输入路径">
    <button id="fb-confirm" class="primary">确认</button>
    <button id="fb-cancel">取消</button>
  </div>
</div>`;
    document.body.appendChild(_modal);

    // 缓存常用元素
    _listEl    = _modal.querySelector('#fb-list');
    _pathEl    = _modal.querySelector('#fb-path-display');
    _confirmBtn = _modal.querySelector('#fb-confirm');

    _modal.querySelector('#fb-close').onclick  = _cancel;
    _modal.querySelector('#fb-cancel').onclick = _cancel;
    _modal.querySelector('#fb-backdrop').onclick = _cancel;
    _modal.querySelector('#fb-up').onclick = () => navigate(_currentPath === '__drives__' ? '' : null, 'parent');
    _confirmBtn.onclick = _confirm;

    // 允许回车确认
    _modal.querySelector('#fb-manual').addEventListener('keydown', e => {
      if (e.key === 'Enter') _confirm();
    });
  }

  // ---- 导航到指定路径 ----
  async function navigate(path, hint) {
    let target = path;
    if (hint === 'parent') {
      // 请求服务器返回父目录
      if (!_currentPath || _currentPath === '__drives__') return;
      try {
        const data = await API._req('GET', `/api/fs/ls?path=${encodeURIComponent(_currentPath)}`);
        target = data.parent;
      } catch (e) { target = ''; }
    }

    try {
      const url = target ? `/api/fs/ls?path=${encodeURIComponent(target)}` : '/api/fs/ls';
      const data = await API._req('GET', url);
      _currentPath = data.path;
      _render(data);
    } catch (e) {
      _listEl.innerHTML = `<div class="fb-error">无法访问: ${e.message}</div>`;
    }
  }

  // ---- 渲染列表 ----
  function _render(data) {
    _currentPath = data.path;
    _pathEl.textContent = data.path === '__drives__' ? '计算机' : data.path;
    _modal.querySelector('#fb-manual').value = data.path === '__drives__' ? '' : data.path;

    _listEl.innerHTML = '';
    if (!data.entries.length) {
      _listEl.innerHTML = '<div class="fb-empty">（空目录）</div>';
      return;
    }
    data.entries.forEach(e => {
      const row = document.createElement('div');
      row.className = 'fb-row' + (e.type === 'file' ? ' fb-file' : '');
      row.innerHTML = `<span class="fb-icon">${e.type === 'dir' ? '📁' : '📄'}</span>
                       <span class="fb-name">${e.name}</span>`;

      if (e.type === 'dir') {
        row.ondblclick = () => navigate(
          data.path === '__drives__' ? e.name : `${data.path}/${e.name}`
        );
        row.onclick = () => {
          _listEl.querySelectorAll('.fb-row').forEach(r => r.classList.remove('selected'));
          row.classList.add('selected');
          if (_mode === 'dir') {
            const dirPath = data.path === '__drives__' ? e.name : `${data.path}/${e.name}`;
            _modal.querySelector('#fb-manual').value = dirPath;
          }
        };
      } else {
        // 文件只在 file 模式下可选
        if (_mode === 'file') {
          row.onclick = () => {
            _listEl.querySelectorAll('.fb-row').forEach(r => r.classList.remove('selected'));
            row.classList.add('selected');
            _modal.querySelector('#fb-manual').value = `${data.path}/${e.name}`;
          };
          row.ondblclick = () => {
            _modal.querySelector('#fb-manual').value = `${data.path}/${e.name}`;
            _confirm();
          };
        }
      }
      _listEl.appendChild(row);
    });
  }

  function _confirm() {
    const val = _modal.querySelector('#fb-manual').value.trim();
    _close();
    if (_resolve && val) _resolve(val);
    _resolve = null;
  }

  function _cancel() {
    _close();
    _resolve = null;
  }

  function _close() {
    if (_modal) _modal.style.display = 'none';
  }

  function _open(mode, title) {
    _ensureModal();
    _mode = mode;
    _modal.querySelector('#fb-title').textContent = title;
    _confirmBtn.textContent = mode === 'dir' ? '选择此目录' : '选择此文件';
    _modal.style.display = 'flex';
    navigate('');   // 从默认根开始
  }

  // ---- 公共 API ----
  return {
    pickDir(callback) {
      return new Promise(res => {
        _resolve = (p) => { callback(p); res(p); };
        _open('dir', '选择目录');
      });
    },
    pickFile(callback) {
      return new Promise(res => {
        _resolve = (p) => { callback(p); res(p); };
        _open('file', '选择文件');
      });
    },
    /**
     * 给 input 元素附加"浏览"按钮。
     * @param {HTMLInputElement} inputEl
     * @param {'dir'|'file'} mode
     */
    attachTo(inputEl, mode = 'dir') {
      // 找到父容器
      const wrap = inputEl.parentElement;
      if (!wrap) return;
      // 如果已经挂了就跳过
      if (wrap.querySelector('.fb-btn')) return;

      const btn = document.createElement('button');
      btn.className = 'fb-btn';
      btn.type = 'button';
      btn.title = mode === 'dir' ? '浏览目录' : '浏览文件';
      btn.textContent = '📂';
      btn.style.cssText = 'flex-shrink:0;padding:5px 8px;';
      btn.onclick = () => {
        const cb = (p) => { inputEl.value = p; inputEl.dispatchEvent(new Event('change')); };
        mode === 'dir' ? FileBrowser.pickDir(cb) : FileBrowser.pickFile(cb);
      };

      // 把 input 和按钮放进一个 flex 行
      wrap.style.display = 'flex';
      wrap.style.gap = '4px';
      wrap.style.alignItems = 'center';
      inputEl.style.flex = '1';
      wrap.appendChild(btn);
    },
  };
})();
