"""Exercise real Git/rsync transfers through a local SSH transport substitute."""
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pie_sync_4090', ROOT / 'sync_4090.py')
sync_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync_tool)


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          stdout=subprocess.PIPE, text=True).stdout.strip()


@pytest.fixture
def endpoints(tmp_path, monkeypatch):
    source, remote = tmp_path / 'local project', tmp_path / 'remote project'
    for repo in (source, remote):
        repo.mkdir()
        git(repo, 'init', '-q')
        (repo / '.gitignore').write_text('docs/\nlogs/\n*.pt\nignored.cache\n.sync_4090/\n')
        (repo / 'policy.py').write_text('before = 123456\n')
        (repo / 'removed.py').write_text('keep remote copy\n')
        git(repo, 'add', '.')
        git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
            'commit', '-qm', 'initial')
    # Independently created commits can differ when this loop crosses a
    # second boundary. Establish the same history, as on a real cloned remote.
    git(remote, 'fetch', '-q', str(source), 'HEAD')
    git(remote, 'reset', '-q', '--hard', 'FETCH_HEAD')
    # The stub executes SSH's remote shell command locally. rsync itself and
    # all Git/file/backup operations are real, including paths with spaces.
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    ssh = bin_dir / 'ssh'
    ssh.write_text('''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
while args and args[0].startswith('-'):
    del args[:2]
os.execv('/bin/sh', ['sh', '-c', ' '.join(args[1:])])
''')
    ssh.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    return source, remote


def args(source, remote, *extra):
    return ['--source', str(source), '--remote-dir', str(remote), '--host', 'local-test', *extra]


def test_sync_uncommitted_files_checksum_backup_exclusions_and_git_history(endpoints):
    source, remote = endpoints
    head = git(remote, 'rev-parse', 'HEAD')
    (source / 'policy.py').write_text('after_ = 654321\n')
    # Equal size and timestamp still require a copy based on content.
    os.utime(source / 'policy.py', ns=((remote / 'policy.py').stat().st_atime_ns,
                                     (remote / 'policy.py').stat().st_mtime_ns))
    (source / 'new config.py').write_text('new untracked config\n')
    (source / 'removed.py').unlink()
    for name in ('docs/note.md', 'logs/training.log', 'model.pt', 'ignored.cache'):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text('local file\n')
        (remote / name).parent.mkdir(parents=True, exist_ok=True)
        (remote / name).write_text('preserve remote file\n')
    # Explicit protections apply even if an artifact was force-added to Git.
    git(source, 'add', '-f', 'docs/note.md', 'model.pt')
    assert sync_tool.main(args(source, remote)) == 0
    assert (remote / 'policy.py').read_text() == 'after_ = 654321\n'
    assert (remote / 'new config.py').read_text() == 'new untracked config\n'
    assert (remote / 'removed.py').read_text() == 'keep remote copy\n'
    assert git(remote, 'rev-parse', 'HEAD') == head
    for name in ('docs/note.md', 'logs/training.log', 'model.pt', 'ignored.cache'):
        assert (remote / name).read_text() == 'preserve remote file\n'
    backups = list((remote / '.sync_4090/backups').glob('*/policy.py'))
    assert len(backups) == 1 and backups[0].read_text() == 'before = 123456\n'
    assert sync_tool.main(args(source, remote)) == 0
    assert len(list((remote / '.sync_4090/backups').glob('*/policy.py'))) == 1


def test_dry_run_does_not_write_remote_files_or_backups(endpoints):
    source, remote = endpoints
    (source / 'policy.py').write_text('preview only\n')
    (source / 'new.py').write_text('preview only\n')
    assert sync_tool.main(args(source, remote, '--dry-run')) == 0
    assert (remote / 'policy.py').read_text() == 'before = 123456\n'
    assert not (remote / 'new.py').exists()
    assert not (remote / '.sync_4090').exists()


def test_external_and_absolute_symlinks_are_not_uploaded(endpoints, tmp_path):
    source, remote = endpoints
    outside = tmp_path / 'private.txt'
    outside.write_text('outside source\n')
    (source / 'external').symlink_to(outside)
    (source / 'absolute').symlink_to(source / 'policy.py')
    (source / 'relative').symlink_to('policy.py')
    assert sync_tool.main(args(source, remote)) == 0
    assert not (remote / 'external').is_symlink()
    assert not (remote / 'absolute').is_symlink()
    assert (remote / 'relative').is_symlink()
    assert os.readlink(remote / 'relative') == 'policy.py'


@pytest.mark.parametrize('invalid', [
    ['--host', '-bad'], ['--host', 'host;echo'], ['--remote-dir', '/'],
    ['--remote-dir', '/home/../'], ['--port', '0'],
])
def test_reject_invalid_destination_arguments(invalid):
    with pytest.raises(SystemExit):
        sync_tool.parse_args(invalid)
