"""Telegram no-op intake reuses the database-versioned working projection."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import ConversationMessage, PartKind
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.transports.telegram_adapter import TelegramBotApp


class IntakeCacheReuseTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.app = TelegramBotApp.__new__(TelegramBotApp)
        self.app.config = replace(self.config, telegram=replace(self.config.telegram,
            keywords=('helper',), ignore_keywords=(), whitelist=()))
        self.app.runtime, self.app.store = self.runtime, self.store
        self.app.artifact_store = ArtifactStore(self.config.artifact_dir)
        self.app.remote_workspace = SimpleNamespace(enabled=False)
        self.app._chat_states = {}
        self.app._ensure_reply_worker = AsyncMock()
        self.app._promote_spontaneous_candidate_after_delay = AsyncMock()
        self.app._notify_user_error = AsyncMock()
        self.bot = SimpleNamespace(id=999, send_message=AsyncMock())
        self.context = SimpleNamespace(bot=self.bot)
        self.chat = SimpleNamespace(id=100, type='supergroup')
        await self.settings(spontaneous_reply_chance=0)

    def update(self, number, text, *, edited=False, photo=None):
        actor = SimpleNamespace(id=7, username='participant', full_name='Participant', is_bot=False)
        message = SimpleNamespace(chat=self.chat, message_id=number, text=text if photo is None else None,
            caption=text if photo else None, from_user=actor, sender_chat=None,
            date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            edit_date=datetime(2026, 1, 2, tzinfo=timezone.utc) if edited else None,
            entities=[], caption_entities=[], reply_to_message=None, message_thread_id=None,
            photo=photo, sticker=None, document=None, animation=None, video=None,
            audio=None, voice=None, video_note=None, reply_text=AsyncMock(), get_bot=lambda: self.bot)
        return SimpleNamespace(effective_chat=self.chat, effective_message=message, effective_user=actor)

    async def group(self, update, *, edited=False):
        handler = self.app.group_edited_message if edited else self.app.group_message
        await handler(update, self.context)
        await asyncio.sleep(0)
        self.app._notify_user_error.assert_not_awaited()

    async def archive(self):
        rows = await self.store.append_messages(self.session,
            [ConversationMessage.user_text('Earlier topic one.'), ConversationMessage.user_text('Earlier topic two.')])
        await self.store.create_memory_block(self.session, source_message_ids=[row.db_id for row in rows],
            summary_text='Two prior topics are retained.', estimated_tokens=12)
        return await self.runtime._get_live_state(self.session)

    async def assert_projection(self, state):
        blocks, messages, version = await self.store.load_live_context_versioned(self.session)
        self.assertEqual(state.database_version, version)
        self.assertEqual(state.blocks, blocks)
        self.assertEqual(state.raw_messages, messages)
        self.assertEqual(state.estimated_tokens,
            sum(block.estimated_tokens for block in blocks) + sum(row.estimated_tokens for row in messages))
        self.assertEqual(state.estimated_images, sum(row.image_count for row in messages))

    async def test_plain_group_intake_and_redelivery_keep_projection_without_archive_reload(self):
        state = await self.archive()
        message = self.update(10, 'Ordinary daily conversation.')
        with patch.object(self.store, 'load_live_context_versioned',
                wraps=self.store.load_live_context_versioned) as loads:
            await self.group(message)
            initial_loads = loads.await_count
            await self.group(message)
        self.assertEqual(initial_loads, 0, 'The unchanged second Telegram envelope must not reload the archive')
        self.assertEqual(loads.await_count, 0, 'Identical redelivery does not change the working projection')
        self.assertEqual(len(state.raw_messages), 1)
        self.assertEqual(state.raw_messages[0].message.metadata['source_revision'], 1)
        self.assertEqual(len(await self.store.list_message_revisions(self.session, state.raw_messages[0].db_id)), 1)
        self.assertEqual(self.provider.requests, [])
        await self.assert_projection(state)

    async def test_media_enrichment_still_reloads_changed_revision_and_decoded_pixels(self):
        state = await self.archive()
        async def download(buffer):
            Image.new('RGB', (2, 2), 'blue').save(buffer, format='PNG')
        file = SimpleNamespace(download_to_memory=AsyncMock(side_effect=download))
        photo = SimpleNamespace(file_id='fixture-photo', file_unique_id='unique-photo', file_size=80,
            width=2, height=2, get_file=AsyncMock(return_value=file))
        with patch.object(self.store, 'load_live_context_versioned',
                wraps=self.store.load_live_context_versioned) as loads:
            await self.group(self.update(10, 'A new photo.', photo=[photo]))
        self.assertGreaterEqual(loads.await_count, 1)
        current = state.raw_messages[0]
        self.assertEqual(current.message.metadata['source_revision'], 2)
        self.assertEqual(current.message.metadata['telegram_intake_stage'], 'complete')
        self.assertEqual(len(await self.store.list_message_revisions(self.session, current.db_id)), 2)
        hydrated = await self.preview_cache.materialize_many(self.session, [current.message], vision=True)
        self.assertTrue(any(part.kind == PartKind.IMAGE and part.data_b64 for part in hydrated[0].parts))
        await self.assert_projection(state)

    async def test_edit_to_compacted_source_restores_current_original_and_invalidates_summary(self):
        await self.group(self.update(10, 'An earlier preference.'))
        state = await self.runtime._get_live_state(self.session)
        source = state.raw_messages[0]
        await self.store.create_memory_block(self.session, source_message_ids=[source.db_id],
            summary_text='The earlier preference.', estimated_tokens=10)
        state = await self.runtime._get_live_state(self.session)
        self.assertEqual(len(state.blocks), 1)
        self.assertEqual(state.raw_messages, [])
        with patch.object(self.store, 'load_live_context_versioned',
                wraps=self.store.load_live_context_versioned) as loads:
            await self.group(self.update(10, 'The corrected preference.', edited=True), edited=True)
        self.assertGreaterEqual(loads.await_count, 1)
        self.assertEqual(state.blocks, [])
        self.assertEqual([row.db_id for row in state.raw_messages], [source.db_id])
        self.assertEqual(state.raw_messages[0].message.metadata['source_revision'], 2)
        self.assertIn('The corrected preference.', '\n'.join(part.text or '' for part in state.raw_messages[0].message.parts))
        await self.assert_projection(state)

    async def test_external_write_during_idempotent_append_requires_projection_reload(self):
        state = await self.archive()
        message = self.update(10, 'Original daily message.')
        await self.group(message)
        append = self.store.append_message
        injected = False
        async def interleaved(*args, **kwargs):
            nonlocal injected
            if not injected:
                injected = True
                await append(self.session, ConversationMessage.user_text('A separate writer committed this original.'))
            return await append(*args, **kwargs)
        with patch.object(self.store, 'load_live_context_versioned',
                wraps=self.store.load_live_context_versioned) as loads, \
                patch.object(self.store, 'append_message', side_effect=interleaved):
            await self.group(message)
        self.assertEqual(loads.await_count, 1, 'Only the actual external projection change requires reconstruction')
        self.assertEqual(len(state.raw_messages), 2)
        self.assertIn('A separate writer committed this original.', state.raw_messages[-1].message.parts[0].text)
        await self.assert_projection(state)
