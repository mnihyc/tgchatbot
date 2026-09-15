"""Build and smoke-test a release bundle using the updater's private file modes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    tag = subprocess.check_output(['tar', '-xOzf', str(bundle), 'RELEASE_TAG'], text=True).strip()
    commit = subprocess.check_output(['tar', '-xOzf', str(bundle), 'RELEASE_COMMIT'], text=True).strip()
    members = subprocess.check_output(['tar', '-tzf', str(bundle)], text=True).splitlines()
    wheels = [member for member in members
              if member.startswith('build/tgchatbot-') and member.endswith('-py3-none-any.whl')]
    assert len(wheels) == 1, 'Release must contain one portable application wheel'
    image = f'tgchatbot-release-smoke:{os.getpid()}'
    (ROOT / 'build').mkdir(exist_ok=True)
    # update.sh uses redirection under umask 077, rather than restoring tar modes.
    # A build from checkout or ordinary tar extraction would hide permission bugs.
    previous_umask = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix='release-smoke-', dir=ROOT / 'build') as directory:
            context = Path(directory)
            for member in ('Dockerfile', '.dockerignore', 'build/runtime-requirements.txt',
                           wheels[0], 'deploy/entrypoint.sh', 'deploy/configure_database.py'):
                destination = context / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('wb') as output:
                    subprocess.run(['tar', '-xOzf', str(bundle), member], stdout=output, check=True)
            assert stat.S_IMODE((context / 'deploy/configure_database.py').stat().st_mode) == 0o600
            subprocess.run(['docker', 'build', '--build-arg', f'RELEASE_TAG={tag}',
                            '--build-arg', f'RELEASE_COMMIT={commit}', '--tag', image,
                            str(context)], check=True)
            subprocess.run(['bash', str(ROOT / 'deploy/smoke-image.sh'), image], check=True)
    finally:
        os.umask(previous_umask)
        subprocess.run(['docker', 'image', 'rm', image], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
