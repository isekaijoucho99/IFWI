"""Publish the delivered additive package from an authenticated LOCAL IFWI clone.

No token is requested or stored. Does not edit the current checkout or main.
Creates a separate worktree at the pinned source commit, checks hashes and
collisions, runs numerical/integration tests, commits new files, optionally pushes.
Run this script from the extracted delivery ZIP, which includes DELIVERY_MANIFEST.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

BASE='226be2262377a0bbde4f813205ceb7cbcd69f335'
BASE_BRANCH='feature/g2ifwi-gpu-batch-tuning'
DEFAULT_BRANCH='feature/ifwi-five-ideas-core-20261030'


def allowed_origin(url: str) -> bool:
    if url.startswith('git@github.com:'): path=url[len('git@github.com:'):]
    else:
        parsed=urlsplit(url)
        if parsed.scheme not in ('https','ssh') or parsed.hostname!='github.com': return False
        path=parsed.path.lstrip('/')
    return path.rstrip('/').removesuffix('.git').lower()=='isekaijoucho99/ifwi'


def validate_delivery(package: Path,worktree: Path) -> list[str]:
    document=json.loads((package/'DELIVERY_MANIFEST.json').read_text(encoding='utf-8'))
    paths=[]
    for item in document['files']:
        name=item['path']; path=PurePosixPath(name)
        if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or '.git' in path.parts:
            raise ValueError('Unsafe delivery path')
        if name in paths: raise ValueError('Duplicate delivery path')
        source=package/name
        if source.is_symlink() or not source.is_file() or package.resolve() not in source.resolve().parents:
            raise ValueError('Invalid source path')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
        if worktree.resolve() not in (worktree/name).resolve().parents:
            raise ValueError('Destination path escapes through a symlink')
        if (worktree/name).exists() or (worktree/name).is_symlink(): raise FileExistsError(f'Refusing to overwrite {name}')
        paths.append(name)
    if not paths: raise ValueError('Empty delivery')
    return paths


def command(arguments,cwd,capture=False):
    return subprocess.run(arguments,cwd=cwd,check=True,text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True,help='Existing authenticated isekaijoucho99/IFWI clone')
    parser.add_argument('--branch',default=DEFAULT_BRANCH)
    parser.add_argument('--worktree',type=Path,help='New directory only; default is a sibling IFWI-five-ideas')
    parser.add_argument('--push',action='store_true',help='Publish ONLY the newly created branch after tests pass')
    parser.add_argument('--dry-run',action='store_true',help='Inspect delivery and origin without fetch/write/commit/push')
    args=parser.parse_args()
    package=Path(__file__).resolve().parents[1]; repo=args.repo.resolve()
    origin=command(['git','remote','get-url','origin'],repo,True).stdout.strip()
    if not allowed_origin(origin): raise ValueError('origin is not isekaijoucho99/IFWI; refusing publication')
    if args.branch in ('main','master',BASE_BRANCH) or not args.branch.startswith('feature/'):
        raise ValueError('Use a NEW feature/ branch, never an existing shared branch')
    command(['git','check-ref-format','--branch',args.branch],repo,True)
    worktree=(args.worktree or repo.parent/(repo.name+'-five-ideas')).resolve()
    if worktree.exists(): raise FileExistsError(f'Worktree destination must not exist: {worktree}')
    paths=validate_delivery(package,worktree)
    print(f'{len(paths)} additive files; base {BASE}; target branch {args.branch}; worktree {worktree}')
    if args.dry_run:
        print('Dry run: no fetch, write, commit or push performed.'); return
    # No credentials or credential configuration are inspected or modified.
    command(['git','fetch','origin',BASE_BRANCH],repo)
    command(['git','cat-file','-e',BASE+'^{commit}'],repo)
    # Refuse overwriting a remotely existing branch even if a non-fast-forward push could be attempted.
    remote=subprocess.run(['git','ls-remote','--exit-code','--heads','origin',args.branch],cwd=repo,text=True,capture_output=True)
    if remote.returncode==0: raise FileExistsError('Remote branch already exists; choose a new branch name')
    if remote.returncode!=2: raise RuntimeError('Cannot check remote branch; publication stopped')
    command(['git','worktree','add','-b',args.branch,str(worktree),BASE],repo)
    paths=validate_delivery(package,worktree)
    for name in paths:
        destination=worktree/name; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(package/name,destination)
    # This suite includes real RNN-FD bridge tests when installed in the checkout.
    command([sys.executable,'-m','pytest','verification/five_ideas','-q'],worktree)
    command([sys.executable,'-m','compileall','-q','ifwi_research','tools/publish_five_ideas.py'],worktree)
    command(['git','add','--',*paths],worktree)
    command(['git','diff','--cached','--check'],worktree)
    # Use the user's existing Git author configuration; do not impersonate them.
    command(['git','commit','-m','feat: add five isolated IFWI research prototypes, protocols and numerical tests'],worktree)
    sha=command(['git','rev-parse','HEAD'],worktree,True).stdout.strip()
    if args.push:
        command(['git','push','-u','origin',args.branch],worktree)
        print(f'Pushed {args.branch}: {sha}')
    else: print(f'Local commit only: {sha}; not pushed. Worktree preserved at {worktree}')

if __name__=='__main__':
    try: main()
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:
        print(f'Publication stopped: {exc}\nNo force push, branch deletion or current-checkout reset was performed.',file=sys.stderr)
        raise SystemExit(1)
