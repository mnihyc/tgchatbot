"""Operator pack descriptions use database configuration, without generation work."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from tgchatbot.stickers import packs
from tgchatbot.storage.sticker_catalog import CatalogConflict


class StickerPackCLITests(unittest.TestCase):
    def setUp(self):
        self.store = SimpleNamespace(pool=SimpleNamespace(open=AsyncMock()), close=AsyncMock())
        self.catalog = SimpleNamespace(
            initialize=AsyncMock(),
            list_pack_descriptions=AsyncMock(return_value={'Blue bird': None, '小狐狸': 'Quiet reactions.'}),
            update_pack_descriptions=AsyncMock(return_value='new-revision'),
        )
        self.enterContext(patch.dict(os.environ, {
            'DATABASE_URL': 'postgresql://fixture:fixture@localhost/fixture',
            'APP_DATA_DIR': str(Path(__file__).parent),
        }, clear=True))
        self.enterContext(patch.object(packs, 'load_dotenv'))
        self.postgres = self.enterContext(patch.object(packs, 'PostgresStore', return_value=self.store))
        self.enterContext(patch.object(packs, 'StickerCatalogStore', return_value=self.catalog))

    def command(self, *argv):
        output = io.StringIO()
        with redirect_stdout(output):
            status = packs.main(list(argv))
        return status, json.loads(output.getvalue())

    def test_list_and_get_distinguish_missing_description_from_unknown_pack_without_credentials(self):
        self.assertEqual(self.command('list'), (0, {'Blue bird': None, '小狐狸': 'Quiet reactions.'}))
        self.assertEqual(self.command('get', 'Blue bird'), (0, {'pack': 'Blue bird', 'description': None}))
        self.assertEqual(self.command('get', '小狐狸'), (0, {'pack': '小狐狸', 'description': 'Quiet reactions.'}))
        self.assertEqual(self.command('get', 'unknown'), (1, {'error': 'Unknown sticker pack: unknown'}))
        self.catalog.update_pack_descriptions.assert_not_awaited()
        self.postgres.assert_called_with('postgresql://fixture:fixture@localhost/fixture')
        self.assertEqual(self.store.close.await_count, 4)

    def test_set_preserves_exact_pack_and_description_and_remove_only_clears_description(self):
        description = '手绘风格。\nGentle, understated reactions.'
        self.assertEqual(self.command('set', '小狐狸', description), (0, {
            'pack': '小狐狸', 'description': description, 'revision_id': 'new-revision',
        }))
        self.catalog.update_pack_descriptions.assert_awaited_once_with({'小狐狸': description})
        self.assertEqual(self.command('remove', '小狐狸'), (0, {
            'pack': '小狐狸', 'description': None, 'revision_id': 'new-revision',
        }))
        self.catalog.update_pack_descriptions.assert_awaited_with({'小狐狸': None})
        self.catalog.list_pack_descriptions.assert_not_awaited()
        self.assertEqual(self.store.close.await_count, 2)

    def test_operator_write_errors_are_reported_without_retrying_or_leaking_connections(self):
        for error in (KeyError('Unknown sticker pack: missing'),
                      ValueError('Pack description must not be blank'),
                      CatalogConflict('Sticker catalog changed; retry with the current revision')):
            with self.subTest(error=type(error).__name__):
                self.catalog.update_pack_descriptions.reset_mock(side_effect=True)
                self.catalog.update_pack_descriptions.side_effect = error
                status, result = self.command('set', 'missing', 'A description')
                self.assertEqual(status, 1)
                self.assertEqual(result, {'error': error.args[0]})
                self.catalog.update_pack_descriptions.assert_awaited_once()
        self.assertEqual(self.store.close.await_count, 3)

    def test_failed_connection_is_closed_and_does_not_attempt_catalog_changes(self):
        self.store.pool.open.side_effect = OSError('Database unavailable')
        self.assertEqual(self.command('remove', 'Blue bird'), (1, {'error': 'Database unavailable'}))
        self.catalog.initialize.assert_not_awaited()
        self.catalog.update_pack_descriptions.assert_not_awaited()
        self.store.close.assert_awaited_once()
