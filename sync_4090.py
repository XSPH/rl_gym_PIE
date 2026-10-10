#!/usr/bin/env python3
"""Sync this Git worktree directly to the 4090; no commit or GitHub required."""
import argparse
from datetime import datetime
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid


PROTECTED_DIRS = {
    '.git', 'docs', 'logs', 'runs', '.conda-envs', '.venv', '__pycache__',
    '.pytest_cache', '.sync_4090',
}
MODEL_SUFFIXES = {'.pt', '.pth', '.ckpt', '.onnx', '.safetensors'}


def selected_files(source):
    """Use Git's ignore rules while including tracked edits and new files."""
    output = subprocess.run(
        ['git', '-C', str(source), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        check=True, stdout=subprocess.PIPE).stdout
    files = []
    for raw_name in sorted(set(output.split(b'\0')) - {b''}):
        name = os.fsdecode(raw_name)
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or set(path.parts) & PROTECTED_DIRS
                or path.suffix.lower() in MODEL_SUFFIXES or path.name == '.env'):
            continue
        local = source / name
        if local.is_symlink():
            if os.path.isabs(os.readlink(local)):
                continue
            try:
                local.resolve().relative_to(source)
            except ValueError:
                continue
        elif not local.is_file():
            continue
        files.append(raw_name)
    return files


def git_text(source, *args):
    return subprocess.run(
        ['git', '-C', str(source), *args], check=True,
        stdout=subprocess.PIPE, text=True).stdout.strip()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='直接同步本地工作区到 4090，包含未提交修改；不提交、不推送、不删除远端文件。')
    parser.add_argument('--host', default='asuka@192.168.1.121', help='SSH 用户和地址')
    parser.add_argument('--remote-dir', default='/home/asuka/rl_gym_PIE_native', help='4090 项目路径')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent, help='本地 Git 项目路径')
    parser.add_argument('--port', type=int, default=22, help='SSH 端口')
    parser.add_argument('--dry-run', action='store_true', help='预览文件变化，不写入远端')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', args.host):
        parser.error('--host 应为主机名、IPv4 地址或 user@host')
    remote = PurePosixPath(args.remote_dir)
    if not remote.is_absolute() or len(remote.parts) < 3 or '..' in remote.parts:
        parser.error('--remote-dir 应为绝对项目路径，不能指向 / 或包含 ..')
    if not 1 <= args.port <= 65535:
        parser.error('--port 应在 1..65535 之间')
    args.remote_dir = str(remote)
    args.source = args.source.expanduser().resolve()
    return args


def sync(args):
    for program in ('git', 'ssh', 'rsync'):
        if shutil.which(program) is None:
            raise RuntimeError('缺少依赖 {}；需要 Git、OpenSSH 和 rsync。'.format(program))
    root = subprocess.run(
        ['git', '-C', str(args.source), 'rev-parse', '--show-toplevel'],
        check=True, stdout=subprocess.PIPE, text=True).stdout.strip()
    if Path(root).resolve() != args.source:
        raise RuntimeError('--source 必须是 Git 项目根目录。')
    branch = git_text(args.source, 'branch', '--show-current')
    if not branch:
        raise RuntimeError('本地仓库处于 detached HEAD；请切换到分支后再同步提交。')
    if subprocess.run(['git', 'check-ref-format', '--branch', branch],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise RuntimeError('本地分支名无效：{}'.format(branch))
    local_head = git_text(args.source, 'rev-parse', 'HEAD')
    files = selected_files(args.source)
    if not files:
        raise RuntimeError('没有可同步的文件。')

    ssh = ['ssh', '-p', str(args.port), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']
    remote_dir = shlex.quote(args.remote_dir)
    preflight = (
        'set -e; command -v rsync >/dev/null; test -d {0}; test -e {0}/.git; '
        'git -C {0} rev-parse --show-toplevel; git -C {0} branch --show-current; '
        'git -C {0} rev-parse HEAD; '
        'if test -n "$(git -C {0} status --porcelain)"; then echo dirty; else echo clean; fi'
    ).format(remote_dir)
    remote_info = subprocess.run(
        ssh + [args.host, preflight], check=True, stdout=subprocess.PIPE,
        text=True).stdout.splitlines()
    if len(remote_info) != 4 or remote_info[0] != args.remote_dir:
        raise RuntimeError('无法读取远端 Git 根目录、分支和工作区状态。')
    remote_branch, remote_head, remote_state = remote_info[1:]
    if remote_branch != branch:
        raise RuntimeError('本地分支 {} 与远端当前分支 {} 不一致；请切换到对应分支。'.format(
            branch, remote_branch or '(detached HEAD)'))
    is_ancestor = subprocess.run(
        ['git', '-C', str(args.source), 'merge-base', '--is-ancestor', remote_head, local_head],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not is_ancestor:
        raise RuntimeError('远端包含本地没有的提交或历史已分叉；为避免覆盖远端提交，已停止同步。')
    commit_count = int(git_text(args.source, 'rev-list', '--count', remote_head + '..' + local_head))
    if commit_count and remote_state != 'clean':
        raise RuntimeError('需要导入 {} 个提交，但远端工作区有未提交修改；请先在 4090 处理这些修改。'.format(
            commit_count))

    label = '预览' if args.dry_run else '同步'
    print('{} {} 个文件：{} → {}:{}'.format(
        label, len(files), args.source, args.host, args.remote_dir), flush=True)
    print('Git 提交：{} 个待快进；工作区文件包含已提交和未提交修改。'.format(commit_count), flush=True)
    print('跳过 Git 忽略文件、docs、日志、模型和环境；不推送 GitHub。', flush=True)
    with tempfile.TemporaryDirectory(prefix='pie-sync-4090-') as temporary:
        manifest = Path(temporary) / 'files'
        manifest.write_bytes(b'\0'.join(files) + b'\0')
        bundle = Path(temporary) / 'commits.bundle'
        incoming = '{}/.sync_4090/incoming/{}-{}.bundle'.format(
            args.remote_dir, branch.replace('/', '_'), uuid.uuid4().hex)
        if commit_count:
            subprocess.run([
                'git', '-C', str(args.source), 'bundle', 'create', str(bundle),
                'refs/heads/' + branch, '^' + remote_head,
            ], check=True)
            if args.dry_run:
                print('Git 提交包约 {}；预览不会传输或导入。'.format(bundle.stat().st_size))
            else:
                subprocess.run(ssh + [args.host, 'mkdir -p {}/.sync_4090/incoming'.format(remote_dir)],
                               check=True)
                bundle_transfer = [
                    'rsync', '--archive', '--no-owner', '--no-group', '--no-times',
                    '--protect-args', '--timeout=30', '-e', shlex.join(ssh),
                    str(bundle), args.host + ':' + incoming,
                ]
                subprocess.run(bundle_transfer, check=True)
                import_commits = (
                    'set -e; git -C {0} fetch --no-tags {1} refs/heads/{2}; '
                    'git -C {0} merge --ff-only FETCH_HEAD; rm -f {1}'
                ).format(remote_dir, shlex.quote(incoming), shlex.quote(branch))
                subprocess.run(ssh + [args.host, import_commits], check=True)
        backup = '{}/.sync_4090/backups/{}'.format(
            args.remote_dir, datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        command = [
            'rsync', '--archive', '--no-owner', '--no-group', '--no-times', '--omit-dir-times',
            '--checksum', '--compress', '--safe-links', '--protect-args', '--timeout=30',
            '--from0', '--files-from=' + str(manifest), '--itemize-changes',
            '-e', shlex.join(ssh), str(args.source) + '/', args.host + ':' + args.remote_dir + '/',
        ]
        if args.dry_run:
            subprocess.run(command[:1] + ['--dry-run'] + command[1:], check=True)
            print('预览结束；未写入远端。')
            return
        subprocess.run(command[:1] + ['--backup', '--backup-dir=' + backup] + command[1:], check=True)
        verification = subprocess.run(
            command[:1] + ['--dry-run'] + command[1:], check=True, stdout=subprocess.PIPE, text=True)
        if verification.stdout.strip():
            print(verification.stdout, end='')
            raise RuntimeError('校验仍有文件差异；请确认本地文件没有同时被修改，再运行一次。')
        print('同步完成，文件内容校验通过。覆盖前的远端文件备份：' + backup)
        print('未删除远端文件；未重启训练。新代码在下次启动或续训时使用。')


def main(argv=None):
    args = parse_args(argv)
    try:
        sync(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print('同步失败：{}'.format(error), file=sys.stderr)
        print('可先检查：ssh -p {} {}'.format(args.port, args.host), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
