"""Local HTTP interface, task concurrency, and real CLI file-sync integration."""
import http.client
import importlib.util
import json
from pathlib import Path
import threading
import time

import pytest

from test_sync_4090 import endpoints, git


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pie_sync_web', ROOT / 'sync_4090_web.py')
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


@pytest.fixture
def server_factory():
    servers = []
    def create(cli=None):
        app = web.SyncApplication(cli or ROOT / 'sync_4090.py')
        server = web.SyncServer(('127.0.0.1', 0), app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server
    yield create
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(server, path, data=None, headers=None):
    client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=8)
    payload = None if data is None else json.dumps(data).encode('utf-8')
    values = {'X-Sync-Token': server.app.token, 'Content-Type': 'application/json'}
    values.update(headers or {})
    client.request('GET' if data is None else 'POST', path, payload, values)
    response = client.getresponse()
    body = response.read()
    result = (response.status, dict(response.getheaders()), body)
    client.close()
    return result


def task_settings(source, remote, action):
    return {'source': str(source), 'remote_dir': str(remote), 'host': 'local-test',
            'port': 22, 'action': action}


def completed(server, task_id):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        status, _, body = request(server, '/api/jobs/' + task_id)
        assert status == 200
        job = json.loads(body)
        if job['status'] != 'running':
            return job
        time.sleep(.03)
    raise AssertionError('Job did not finish within 8 seconds')


def test_serves_local_ui_assets_and_blocks_other_hosts_and_paths(server_factory):
    server = server_factory()
    status, headers, body = request(server, '/')
    assert status == 200 and 'text/html' in headers['Content-Type']
    assert server.app.token.encode() in body and b'__SYNC_TOKEN__' not in body
    assert headers['X-Frame-Options'] == 'DENY'
    for path, mime in (('/app.js', 'javascript'), ('/style.css', 'text/css')):
        status, headers, body = request(server, path)
        assert status == 200 and mime in headers['Content-Type'] and body
    assert request(server, '/../../sync_4090.py')[0] == 404
    assert request(server, '/', headers={'Host': 'untrusted.example'})[0] == 403


def test_http_preview_parses_file_changes_and_keeps_remote_unchanged(server_factory, endpoints):
    source, remote = endpoints
    (source / 'policy.py').write_text('modified through UI\n')
    (source / 'new settings').mkdir()
    (source / 'new settings/config.py').write_text('new file\n')
    server = server_factory()
    status, _, body = request(server, '/api/jobs', task_settings(source, remote, 'preview'))
    assert status == 202
    job = completed(server, json.loads(body)['id'])
    assert job['status'] == 'success' and job['candidate_count'] == 4
    assert {change['path']: change['kind'] for change in job['changes']} == {
        'policy.py': 'modified', 'new settings/config.py': 'added'}
    assert (remote / 'policy.py').read_text() == 'before = 123456\n'
    assert not (remote / 'new settings').exists()
    assert not (remote / '.sync_4090').exists()
    history = json.loads(request(server, '/api/jobs')[2])
    assert history[0]['id'] == job['id'] and 'logs' not in history[0]


def test_http_sync_uses_real_cli_backup_and_checksum(server_factory, endpoints):
    source, remote = endpoints
    head = git(remote, 'rev-parse', 'HEAD')
    (source / 'policy.py').write_text('copied through UI\n')
    server = server_factory()
    status, _, body = request(server, '/api/jobs', task_settings(source, remote, 'sync'))
    assert status == 202
    job = completed(server, json.loads(body)['id'])
    assert job['status'] == 'success' and job['returncode'] == 0
    assert (remote / 'policy.py').read_text() == 'copied through UI\n'
    assert (Path(job['backup']) / 'policy.py').read_text() == 'before = 123456\n'
    assert git(remote, 'rev-parse', 'HEAD') == head


def test_post_requires_page_token_and_same_origin(server_factory, endpoints):
    source, remote = endpoints
    server = server_factory()
    data = task_settings(source, remote, 'sync')
    assert request(server, '/api/jobs', data, {'X-Sync-Token': 'bad'})[0] == 403
    assert request(server, '/api/jobs', data, {'Origin': 'https://untrusted.example'})[0] == 403
    assert server.app.snapshot() == []


@pytest.mark.parametrize('field,value', [
    ('action', 'unknown'), ('host', 'host;touch /tmp/unsafe'),
    ('remote_dir', '/'), ('port', '22'), ('port', True), ('source', '/missing/source'),
])
def test_invalid_settings_never_start_a_process(server_factory, endpoints, field, value):
    source, remote = endpoints
    server = server_factory()
    data = task_settings(source, remote, 'preview')
    data[field] = value
    assert request(server, '/api/jobs', data)[0] == 400
    assert server.app.snapshot() == []


def test_one_job_at_a_time_and_failed_process_is_visible(server_factory, endpoints, tmp_path):
    source, remote = endpoints
    failing = tmp_path / 'slow_cli.py'
    failing.write_text('import time,sys\nprint("SSH connection failed",flush=True)\ntime.sleep(.3)\nsys.exit(1)\n')
    server = server_factory(failing)
    data = task_settings(source, remote, 'preview')
    status, _, body = request(server, '/api/jobs', data)
    assert status == 202
    assert request(server, '/api/jobs', data)[0] == 409
    job = completed(server, json.loads(body)['id'])
    assert job['status'] == 'error' and job['returncode'] == 1
    assert 'SSH connection failed' in job['logs']
    assert request(server, '/api/jobs/not-a-job')[0] == 404
