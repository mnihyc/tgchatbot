"""Installed maintenance commands use the operator's installation configuration."""
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

from scripts import query_sticker_index
from tgchatbot.stickers import build, manage, packs
from tgchatbot.tools import memory


class ConfigurationLoaded(Exception):
    pass


class MaintenanceConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_operator_directory_owns_env_and_explicit_environment_keeps_precedence(self):
        with TemporaryDirectory(prefix='fixture-maintenance-', dir=Path(__file__).parent) as temporary:
            root = Path(temporary).resolve()
            (root / '.env').write_text('DATABASE_URL=postgresql://parent/incorrect\n')
            deployment, empty = root / 'installation', root / 'empty-installation'
            deployment.mkdir()
            empty.mkdir()
            (deployment / '.env').write_text('DATABASE_URL=postgresql://installation/correct\n')

            def stop_before_database_or_providers(**kwargs):
                observed.append(os.getenv('DATABASE_URL'))
                raise ConfigurationLoaded()

            commands = [(module.run, f'{module.__name__}.load_config') for module in
                        (packs, manage, build, query_sticker_index)]
            commands.append((memory._run, 'tgchatbot.config.load_config'))
            for directory, inherited, expected in (
                (deployment, {}, 'postgresql://installation/correct'),
                (deployment, {'DATABASE_URL': 'postgresql://environment/override'}, 'postgresql://environment/override'),
                (empty, {}, None),
            ):
                for run, target in commands:
                    observed = []
                    with self.subTest(command=run.__module__, directory=directory.name, inherited=bool(inherited)), \
                         chdir(directory), patch.dict(os.environ, inherited, clear=True), \
                         patch(target, side_effect=stop_before_database_or_providers):
                        with self.assertRaises(ConfigurationLoaded):
                            await run(SimpleNamespace())
                    self.assertEqual(observed, [expected])
