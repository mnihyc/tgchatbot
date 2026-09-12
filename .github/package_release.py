"""Create the deployment release bundle."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def run(*command: str, cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build_wheel(build: Path) -> None:
    # setuptools may reuse build/lib from earlier versions. A release owns a
    # fresh build tree containing only current tracked application inputs.
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--',
        'pyproject.toml', 'README.md', 'LICENSE', 'tgchatbot', 'scripts'], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix='release-source-', dir=build) as directory:
        source = Path(directory)
        for name in tracked.decode().split('\0'):
            if not name:
                continue
            original = ROOT / name
            if original.is_symlink():
                raise ValueError(f'Release source must be a regular repository file: {name}')
            if not original.is_file():
                continue  # A tracked deletion in the current working tree.
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
        run('uv', 'build', '--wheel', '--out-dir', str(build), cwd=source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', help='Release tag; defaults to the project version')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare wheel and locked Docker dependency input')
    args = parser.parse_args()
    version = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']
    tag = args.tag or 'v' + version
    if tag != 'v' + version:
        parser.error('Release tag must match the project version')
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    run('uv', 'export', '--frozen', '--no-dev', '--no-emit-project', '--no-header',
        '--no-annotate', '--output-file', 'build/runtime-requirements.txt')
    build_wheel(build)
    wheel = build / f'tgchatbot-{version}-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as package:
        for name in package.namelist():
            top = name.split('/', 1)[0]
            if top not in {'tgchatbot', 'scripts', f'tgchatbot-{version}.dist-info'}:
                raise ValueError(f'Unexpected application wheel content: {name}')
            if name.endswith(('.pyc', '.so', '.pyd')) or '__pycache__' in name.split('/'):
                raise ValueError(f'Application wheel must contain portable code: {name}')
    if args.prepare_only:
        print(f'Prepared {wheel.name} ({wheel.stat().st_size:,} bytes)')
        return
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dist = ROOT / 'dist'
    dist.mkdir(exist_ok=True)
    bundle = dist / f'tgchatbot-deploy-{tag}.tar.gz'
    members = {
        'compose.yml': 'deploy/compose.yml',
        'update.sh': 'deploy/update.sh',
        'README.md': 'deploy/README.md',
        '.env.example': '.env.example',
        '.env.full.example': '.env.full.example',
        'LICENSE': 'LICENSE',
        'Dockerfile': 'Dockerfile',
        '.dockerignore': '.dockerignore',
        'build/runtime-requirements.txt': 'build/runtime-requirements.txt',
        'build/' + wheel.name: 'build/' + wheel.name,
        'deploy/entrypoint.sh': 'deploy/entrypoint.sh',
        'deploy/configure_database.py': 'deploy/configure_database.py',
    }
    with tarfile.open(bundle, 'w:gz') as archive:
        for destination, source in members.items():
            archive.add(ROOT / source, arcname=destination, recursive=False)
        # Required by already shipped updaters before they hand off to this one.
        for name, value in {'RELEASE_TAG': tag, 'RELEASE_COMMIT': commit}.items():
            data = (value + '\n').encode()
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    print(f'Packaged {bundle.name} ({bundle.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
