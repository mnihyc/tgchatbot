"""Create the deployment release bundle."""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


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
    run('uv', 'build', '--wheel', '--out-dir', 'build')
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
    updater = dist / 'update.sh'
    shutil.copy2(ROOT / 'deploy/update.sh', updater)
    lines = []
    for artifact in (bundle, updater):
        with artifact.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        lines.append(f'{digest}  {artifact.name}\n')
    (dist / 'SHA256SUMS').write_text(''.join(lines))
    print(f'Packaged {bundle.name} ({bundle.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
