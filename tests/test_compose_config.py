"""Render real Compose configuration without starting services or using secrets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class ComposeConfigurationTests(unittest.TestCase):
    def test_bot_env_values_and_service_mounts_survive_real_compose_parsing(self):
        docker = shutil.which('docker')
        if docker is None:
            self.skipTest('Docker Compose is unavailable')
        # Retain CLI discovery settings, without inheriting application secrets
        # or interpolation values from the developer's shell environment.
        environment = {
            key: value for key, value in os.environ.items()
            if key in {'PATH', 'HOME', 'DOCKER_CONFIG', 'DOCKER_CLI_PLUGIN_EXTRA_DIRS'}
        }
        try:
            available = subprocess.run(
                [docker, 'compose', 'version'], cwd=REPO, env=environment,
                capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            self.skipTest('Docker Compose is unavailable')
        if available.returncode != 0:
            self.skipTest('Docker Compose is unavailable')

        with tempfile.TemporaryDirectory(prefix='fixture-compose-', dir=REPO / 'tests') as temporary:
            root = Path(temporary)
            shutil.copyfile(REPO / 'deploy' / 'compose.yml', root / 'compose.yml')
            token = '123456:synthetic-telegram-token'
            key = 'synthetic-provider-key'
            prompt = 'First line costs $5; keep $PROMPT_TOKEN literally.\nSecond line keeps ${PROMPT_TOKEN} and "quotes".'
            (root / '.env').write_text(
                f'TGBOT_TOKEN={token}\n'
                f'OPENAI_API_KEY={key}\n'
                f"DEFAULT_SYSTEM_PROMPT='{prompt}'\n"
                'APP_DATA_DIR=/legacy/operator/data\n',
                encoding='utf-8',
            )
            # This is a local config renderer: no daemon, build, pull, or service
            # startup is involved. Bound CLI execution in case plugin startup hangs.
            rendered = subprocess.run(
                [docker, 'compose', '--file', 'compose.yml', 'config', '--format', 'json'],
                cwd=root, env=environment, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(rendered.returncode, 0, 'Compose could not render the synthetic fixture')
            services = json.loads(rendered.stdout)['services']
            bot_environment = services['bot']['environment']
            self.assertEqual(bot_environment['TGBOT_TOKEN'], token)
            self.assertEqual(bot_environment['OPENAI_API_KEY'], key)
            # `config` serializes literal dollars as $$ so its canonical output
            # can be fed back into Compose without expanding the prompt again.
            self.assertEqual(bot_environment['DEFAULT_SYSTEM_PROMPT'], prompt.replace('$', '$$'))
            self.assertEqual(bot_environment['APP_DATA_DIR'], '/app/data')
            self.assertEqual(bot_environment['APP_TEMP_DIR'], '/tmp')

            retriever_environment = services['retriever']['environment']
            for secret_name in ('TGBOT_TOKEN', 'OPENAI_API_KEY', 'DEFAULT_SYSTEM_PROMPT'):
                self.assertNotIn(secret_name, retriever_environment)
            for service_name in ('bot', 'retriever'):
                volumes = services[service_name]['volumes']
                bind_mounts = {volume['target']: volume['source'] for volume in volumes if volume['type'] == 'bind'}
                self.assertEqual(bind_mounts['/app/data'], str(root / 'data'))
                self.assertEqual(bind_mounts['/tmp'], str(root / 'tmp' / service_name))
                self.assertFalse(any(Path(volume['source']).name == '.env' or Path(volume['target']).name == '.env' for volume in volumes))
