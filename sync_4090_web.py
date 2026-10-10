#!/usr/bin/env python3
"""Local web interface for sync_4090.py, using only the Python standard library."""
import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
from urllib.parse import urlsplit
import webbrowser


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'sync_ui'


class SyncApplication:
    def __init__(self, cli_path=ROOT / 'sync_4090.py'):
        self.cli_path = Path(cli_path)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.jobs = {}
        self.active = None
        self.process = None
        self.closing = False

    def configuration(self):
        with self.lock:
            return {'host': 'asuka@192.168.1.121', 'remote_dir': '/home/asuka/rl_gym_PIE_native',
                    'source': str(ROOT), 'port': 22, 'active': self.active}

    def start(self, data):
        action = data.get('action')
        if action not in ('preview', 'sync'):
            raise ValueError('请选择预览或同步。')
        host = data.get('host', '')
        remote = data.get('remote_dir', '')
        source = data.get('source', str(ROOT))
        port = data.get('port', 22)
        if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', host):
            raise ValueError('SSH 地址应为 user@IP 或主机名。')
        if (not isinstance(remote, str) or not remote.startswith('/')
                or len(Path(remote).parts) < 3 or '..' in Path(remote).parts):
            raise ValueError('远端目录应为绝对项目路径。')
        if not isinstance(source, str) or not Path(source).expanduser().is_dir():
            raise ValueError('本地项目目录不存在。')
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('SSH 端口应在 1..65535 之间。')
        settings = {'host': host, 'remote_dir': remote,
                    'source': str(Path(source).expanduser().resolve()), 'port': port}
        command = [sys.executable, '-u', str(self.cli_path), '--host', host,
                   '--remote-dir', remote, '--source', settings['source'], '--port', str(port)]
        if action == 'preview':
            command.append('--dry-run')
        with self.lock:
            if self.active or self.closing:
                raise RuntimeError('已有任务正在运行，请等它完成。')
            job_id = secrets.token_hex(8)
            job = {'id': job_id, 'action': action, 'settings': settings, 'status': 'running',
                   'started': datetime.now().isoformat(timespec='seconds'), 'finished': None,
                   'logs': [], 'changes': [], 'candidate_count': None, 'backup': None,
                   'commit_count': 0, 'returncode': None}
            self.jobs[job_id] = job
            self.active = job_id
            while len(self.jobs) > 20:
                del self.jobs[next(iter(self.jobs))]
        threading.Thread(target=self._run, args=(job_id, command), daemon=True).start()
        return job_id

    def _run(self, job_id, command):
        returncode = 1
        try:
            with self.lock:
                if self.closing:
                    raise RuntimeError('网页服务正在关闭。')
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace', bufsize=1,
                                           start_new_session=True)
                self.process = process
            for line in process.stdout:
                line = line.rstrip('\r\n')[:4096]
                with self.lock:
                    job = self.jobs[job_id]
                    if len(job['logs']) < 5000:
                        job['logs'].append(line)
                    match = re.match(r'^(?:预览|同步) (\d+) 个文件', line)
                    if match:
                        job['candidate_count'] = int(match.group(1))
                    match = re.match(r'^Git 提交：(\d+) 个待快进', line)
                    if match:
                        job['commit_count'] = int(match.group(1))
                    change = re.match(r'^([<>ch.][fdLDS]\S{9}) (.+)$', line)
                    if change and change.group(1)[1] != 'd':
                        code, name = change.groups()
                        kind = 'added' if '+++++++++' in code else 'metadata' if code[0] == '.' else 'modified'
                        job['changes'].append({'path': name, 'kind': kind})
                    if '覆盖前的远端文件备份：' in line:
                        job['backup'] = line.split('覆盖前的远端文件备份：', 1)[1]
            returncode = process.wait()
        except Exception as error:
            with self.lock:
                self.jobs[job_id]['logs'].append('任务失败：' + str(error))
        finally:
            with self.lock:
                job = self.jobs[job_id]
                job['returncode'] = returncode
                job['status'] = 'success' if returncode == 0 else 'error'
                job['finished'] = datetime.now().isoformat(timespec='seconds')
                self.active = None
                self.process = None

    def snapshot(self, job_id=None):
        with self.lock:
            if job_id is not None:
                job = self.jobs.get(job_id)
                return json.loads(json.dumps(job)) if job else None
            return [{key: value for key, value in job.items() if key not in ('logs', 'changes')}
                    for job in reversed(list(self.jobs.values()))]

    def close(self):
        with self.lock:
            self.closing = True
            process = self.process
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


class SyncServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        self.app = app
        super().__init__(address, Handler)

    def server_close(self):
        self.app.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def respond(self, status, body, content_type='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def allowed_host(self):
        port = self.server.server_port
        if self.headers.get('Host') not in ('127.0.0.1:' + str(port), 'localhost:' + str(port)):
            self.respond(403, {'error': '仅允许从本机地址访问。'})
            return False
        return True

    def do_GET(self):
        if not self.allowed_host():
            return
        path = urlsplit(self.path).path
        if path == '/api/config':
            self.respond(200, self.server.app.configuration())
        elif path == '/api/jobs':
            self.respond(200, self.server.app.snapshot())
        elif path.startswith('/api/jobs/'):
            job = self.server.app.snapshot(path.rsplit('/', 1)[-1])
            self.respond(200 if job else 404, job or {'error': '任务不存在。'})
        elif path in ('/', '/app.js', '/style.css'):
            filename, mime = {'/': ('index.html', 'text/html; charset=utf-8'),
                              '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                              '/style.css': ('style.css', 'text/css; charset=utf-8')}[path]
            body = (ASSETS / filename).read_bytes()
            if path == '/':
                body = body.replace(b'__SYNC_TOKEN__', self.server.app.token.encode('ascii'))
            self.respond(200, body, mime)
        elif path == '/favicon.ico':
            self.respond(204, b'')
        else:
            self.respond(404, {'error': '页面不存在。'})

    def do_POST(self):
        if not self.allowed_host():
            return
        origin = self.headers.get('Origin')
        if origin and origin not in ('http://127.0.0.1:' + str(self.server.server_port),
                                     'http://localhost:' + str(self.server.server_port)):
            self.respond(403, {'error': '请求来源不匹配，请从本地网页操作。'})
            return
        if self.headers.get('X-Sync-Token') != self.server.app.token:
            self.respond(403, {'error': '页面已过期，请刷新后再试。'})
            return
        if urlsplit(self.path).path != '/api/jobs':
            self.respond(404, {'error': '接口不存在。'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8192:
                raise ValueError('请求内容为空或过长。')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('请求格式错误。')
            job_id = self.server.app.start(data)
            self.respond(202, {'id': job_id})
        except (ValueError, TypeError) as error:
            self.respond(400, {'error': str(error)})
        except RuntimeError as error:
            self.respond(409, {'error': str(error)})


def main():
    parser = argparse.ArgumentParser(description='打开 4090 同步工作台，无需额外 Python 依赖。')
    parser.add_argument('--port', type=int, default=8765, help='本地网页端口，默认 8765')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('--port 应在 0..65535 之间')
    try:
        server = SyncServer(('127.0.0.1', args.port), SyncApplication())
    except OSError as error:
        print('无法启动网页：{}；可用 --port 8766 换一个端口。'.format(error), file=sys.stderr)
        return 1
    url = 'http://127.0.0.1:{}'.format(server.server_port)
    print('4090 同步工作台：' + url, flush=True)
    print('按 Ctrl+C 关闭。', flush=True)
    if not args.no_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
