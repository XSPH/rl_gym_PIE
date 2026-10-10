'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="api-token"]').content;
let ready = false, busy = false, currentJob = null, pollTimer = null, toastTimer = null;
const fields = ['host', 'remote-dir', 'source', 'ssh-port'];

function toast(message) {
  $('toast').textContent = message;
  $('toast').classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('toast').classList.add('hidden'), 4200);
}
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {
    ...(options.body ? {'Content-Type': 'application/json', 'X-Sync-Token': token} : {}), ...options.headers
  }});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
function settings() {
  return {host: $('host').value.trim(), remote_dir: $('remote-dir').value.trim(),
    source: $('source').value.trim(), port: Number($('ssh-port').value)};
}
function fillSettings(data) {
  $('host').value = data.host;
  $('remote-dir').value = data.remote_dir;
  $('source').value = data.source;
  $('ssh-port').value = data.port;
  updateRoute();
}
function updateRoute() {
  const data = settings();
  for (const [id, value] of [['host-summary', data.host], ['remote-summary', data.remote_dir], ['source-summary', data.source]]) {
    $(id).textContent = value || '尚未填写';
    $(id).title = value;
  }
}
function setBusy(value) {
  busy = value;
  fields.forEach(id => $(id).disabled = value);
  $('preview-button').disabled = value || !ready;
  $('sync-button').disabled = value || !ready;
  document.querySelectorAll('.history-open').forEach(button => button.disabled = value);
}
function showView(view) {
  ['workspace', 'history', 'guide'].forEach(name => $(name + '-view').classList.toggle('hidden', name !== view));
  document.querySelectorAll('[data-view]').forEach(button => button.classList.toggle('active', button.dataset.view === view));
  $('breadcrumb').textContent = {workspace: '同步工作台', history: '运行记录', guide: '使用说明'}[view];
  if (view === 'history') loadHistory();
}
function emptyFiles(title, message) {
  $('file-list').replaceChildren();
  const box = document.createElement('div'); box.className = 'empty-state';
  const icon = document.createElement('span'); icon.textContent = '▱';
  const heading = document.createElement('strong'); heading.textContent = title;
  const description = document.createElement('p'); description.textContent = message;
  box.append(icon, heading, description); $('file-list').append(box);
}
function resetResult() {
  currentJob = null;
  ['changed-count', 'added-count', 'modified-count', 'candidate-count'].forEach(id => $(id).textContent = '—');
  $('run-state').className = 'run-state'; $('state-icon').textContent = '◎';
  $('state-title').textContent = '准备好同步';
  $('state-description').textContent = '点击「预览变化」，检查将要更新的文件。';
  $('connection-state').textContent = '尚未检测'; $('connection-state').className = 'pill';
  $('backup-note').classList.add('hidden');
  emptyFiles('先看变化，再开始同步', '预览会比较文件内容，不会修改远端文件。');
}
function renderJob(job) {
  currentJob = job;
  const changes = job.changes || [], added = changes.filter(change => change.kind === 'added').length;
  $('changed-count').textContent = changes.length;
  $('added-count').textContent = added;
  $('modified-count').textContent = changes.length - added;
  $('candidate-count').textContent = job.candidate_count ?? '—';
  const running = job.status === 'running', success = job.status === 'success', preview = job.action === 'preview';
  $('result-mode').textContent = preview ? 'PREVIEW' : 'SYNC';
  $('run-state').className = 'run-state ' + (running ? 'running' : success ? 'success' : 'error');
  $('state-icon').textContent = running ? '◌' : success ? '✓' : '!';
  $('state-title').textContent = running ? (preview ? '正在预览文件变化' : '正在同步与校验') :
    success ? (preview ? '预览完成' : '同步完成，内容校验通过') : (preview ? '预览失败' : '同步失败');
  const commits = job.commit_count || 0;
  $('state-description').textContent = running ? (job.logs.length ? '正在执行，请等待结果…' : '正在通过 SSH 连接 4090…') :
    success ? (preview ? `发现 ${changes.length} 个变化文件，${commits} 个本地提交待导入；远端尚未修改。` : `已导入 ${commits} 个提交；备份已保留，下次启动或续训会使用新源码。`) :
    '查看下方日志，检查 SSH 连接、项目路径和依赖。';
  $('connection-state').textContent = running ? '连接中' : success ? '连接正常' : '执行失败';
  $('connection-state').className = 'pill ' + (success ? 'connected' : running ? '' : 'failed');
  if (changes.length) {
    $('file-list').replaceChildren();
    for (const change of changes) {
      const row = document.createElement('div'); row.className = 'file-row';
      const icon = document.createElement('span'); icon.className = 'file-icon'; icon.textContent = '▧';
      const path = document.createElement('span'); path.className = 'file-name'; path.textContent = change.path; path.title = change.path;
      const badge = document.createElement('span'); badge.className = 'change-badge ' + change.kind;
      badge.textContent = {added: '新增', modified: '修改', metadata: '属性'}[change.kind];
      row.append(icon, path, badge); $('file-list').append(row);
    }
  } else emptyFiles(running ? '正在比较文件…' : success ? '本地与远端文件一致' : '尚未取得文件变化',
    running ? '文件变化会在这里实时显示。' : success ? '本次没有需要传输的文件。' : '详细原因见执行日志。');
  const output = $('log-output'), nearBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 50;
  output.textContent = job.logs.join('\n') || '正在连接远端，请稍候…';
  if (nearBottom) output.scrollTop = output.scrollHeight;
  $('backup-note').classList.toggle('hidden', !job.backup);
  $('backup-path').textContent = job.backup || '';
}
async function watchJob(id) {
  clearTimeout(pollTimer);
  try {
    const job = await api('/api/jobs/' + id);
    renderJob(job); setBusy(job.status === 'running');
    if (job.status === 'running') pollTimer = setTimeout(() => watchJob(id), 700);
  } catch (error) {
    ready = false; setBusy(false);
    $('state-title').textContent = '与本地服务的连接中断';
    $('state-description').textContent = '确认网页服务仍在运行，然后刷新页面。';
    $('run-state').className = 'run-state error'; toast(error.message);
  }
}
async function run(action) {
  if (busy || !ready || !$('settings-form').reportValidity()) return;
  const data = settings();
  try { localStorage.setItem('pie-sync-settings', JSON.stringify(data)); } catch (_) { /* Private browsing may disable storage. */ }
  setBusy(true); resetResult();
  $('run-state').className = 'run-state running'; $('state-title').textContent = '正在创建任务';
  try {
    const job = await api('/api/jobs', {method: 'POST', body: JSON.stringify({...data, action})});
    await watchJob(job.id);
  } catch (error) {
    setBusy(false); $('run-state').className = 'run-state error'; $('state-icon').textContent = '!';
    $('state-title').textContent = '任务未启动'; $('state-description').textContent = error.message; toast(error.message);
  }
}
async function loadHistory() {
  try {
    const jobs = await api('/api/jobs');
    $('history-list').replaceChildren();
    if (!jobs.length) {
      const empty = document.createElement('div'); empty.className = 'empty-state';
      const title = document.createElement('strong'); title.textContent = '还没有运行记录';
      const description = document.createElement('p'); description.textContent = '完成一次预览或同步后，会在这里显示。';
      empty.append(title, description); $('history-list').append(empty); return;
    }
    for (const job of jobs) {
      const row = document.createElement('div'); row.className = 'history-row';
      const info = document.createElement('div'); info.className = 'history-info';
      const title = document.createElement('strong'); title.textContent = job.action === 'preview' ? '预览文件变化' : '同步工作区';
      const host = document.createElement('p'); host.textContent = job.settings.host + ' · ' + job.settings.remote_dir;
      const time = document.createElement('span'); time.className = 'history-time'; time.textContent = job.started.replace('T', ' ');
      info.append(title, host, time);
      const status = document.createElement('span'); status.className = 'history-status ' + job.status;
      status.textContent = {running: '运行中', success: '完成', error: '失败'}[job.status];
      const open = document.createElement('button'); open.className = 'text-button history-open'; open.textContent = '查看 →'; open.disabled = busy;
      open.addEventListener('click', () => { fillSettings(job.settings); showView('workspace'); watchJob(job.id); });
      row.append(info, status, open); $('history-list').append(row);
    }
  } catch (error) { toast(error.message); }
}
$('preview-button').addEventListener('click', () => run('preview'));
$('sync-button').addEventListener('click', () => run('sync'));
$('settings-form').addEventListener('submit', event => { event.preventDefault(); run('preview'); });
fields.forEach(id => $(id).addEventListener('input', () => { updateRoute(); if (!busy) resetResult(); }));
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => showView(button.dataset.view)));
$('refresh-history').addEventListener('click', loadHistory);
$('copy-log').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('log-output').textContent); toast('日志已复制'); }
  catch (_) { toast('无法自动复制，可直接选中日志文字复制。'); }
});
async function initialize() {
  setBusy(false);
  try {
    const config = await api('/api/config');
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem('pie-sync-settings')) || {}; } catch (_) {}
    fillSettings({...config, ...Object.fromEntries(Object.entries(saved).filter(([key]) => ['host', 'remote_dir', 'source', 'port'].includes(key)))});
    ready = true; setBusy(false);
    const history = await api('/api/jobs');
    const lastId = config.active || history[0]?.id;
    if (lastId) {
      const job = await api('/api/jobs/' + lastId); fillSettings(job.settings); await watchJob(lastId);
    }
  } catch (error) {
    $('state-title').textContent = '本地服务不可用'; $('state-description').textContent = '请启动 sync_4090_web.py 后刷新页面。';
    $('run-state').className = 'run-state error'; toast(error.message);
  }
}
initialize();
