"""Retained visual evidence follows original-message ownership and reset scope."""
from __future__ import annotations

import base64
import io
import json

from PIL import Image

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.storage.postgres_store import PostgresStore, StaleScopeError
from tgchatbot.storage.previews import PreviewCache


def pixels(color):
    output = io.BytesIO()
    with Image.new('RGB', (2, 2), color) as image:
        image.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode('ascii')


class MessageImageWorkflows(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.red, self.blue = pixels('red'), pixels('blue')

    async def original(self, number, *, session=None, actor=101, parts=None, text='Image context.'):
        session = session or self.session
        await self.store.get_or_create_session(session, self.config.default_session_settings())
        return await self.store.append_message(session, ConversationMessage(MessageRole.USER,
            parts if parts is not None else [MessagePart(PartKind.TEXT, text=text),
                MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=self.red)],
            metadata={'source': 'telegram', 'source_chat_id': session.split(':', 1)[1],
                'source_message_id': str(number), 'actor_id': f'telegram:user:{actor}',
                'actor_kind': 'user', 'actor_name': 'Shared display name',
                'sent_at': '2026-01-01T00:00:00+00:00'}))

    async def descriptors(self, *sources, session=None):
        return await self.store.describe_message_images(session or self.session, [row.db_id for row in sources])

    async def test_ordered_frames_keep_their_own_source_and_missing_image_placeholder(self):
        first = await self.original(1, actor=101)
        second = await self.original(2, actor=102, parts=[
            MessagePart(PartKind.STICKER, text='🙂'),
            MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=self.blue),
            MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=self.red)])
        missing = await self.original(3, parts=[MessagePart(PartKind.TEXT, text='Older exported photo.'),
            MessagePart(PartKind.IMAGE, filename='missing.jpg', mime_type='image/jpeg',
                origin='attachment_reference', detail='Image bytes were not imported.')])
        described = await self.descriptors(first, second, missing)
        self.assertEqual(described[first.db_id], [{'image_id': f'img:{first.db_id}:1:1',
            'mime_type': 'image/png', 'available': True}])
        self.assertEqual([item['image_id'] for item in described[second.db_id]],
            [f'img:{second.db_id}:1:1', f'img:{second.db_id}:1:2'])
        self.assertFalse(described[missing.db_id][0]['available'])
        self.assertNotIn('data_b64', json.dumps(described))
        requested = [described[second.db_id][1]['image_id'], described[first.db_id][0]['image_id'],
            described[missing.db_id][0]['image_id']]
        result = await self.store.resolve_message_images(self.session,
            [first.db_id, second.db_id, missing.db_id], requested)
        self.assertEqual([item['image_id'] for item in result['image_results']], requested)
        self.assertEqual([item['status'] for item in result['image_results']], ['selected', 'selected', 'unavailable'])
        self.assertEqual([part.data_b64 for part in result['evidence_parts'] if part.kind == PartKind.IMAGE],
            [self.red, self.red])
        labels = [part.text for part in result['evidence_parts'] if part.kind == PartKind.TEXT]
        self.assertIn('telegram:user:102', labels[0])
        self.assertIn(f'"message_id": {second.db_id}', labels[0])
        self.assertIn('telegram:user:101', labels[1])
        self.assertEqual([part.origin for part in result['evidence_parts']],
            [f'memory_image:{requested[0]}'] * 2 + [f'memory_image:{requested[1]}'] * 2)

    async def test_prompt_retirement_episode_and_soft_reset_keep_cold_recall_bytes(self):
        source = await self.original(1)
        image_id = (await self.descriptors(source))[source.db_id][0]['image_id']
        reference = source.message.parts[-1].preview_ref
        canonical = (await self.store.read_messages(self.session, [source.db_id]))[0].message
        await self.store.create_memory_block(self.session, source_message_ids=[source.db_id],
            summary_text='The original photo was discussed.', estimated_tokens=8)
        self.assertEqual((await self.store.resolve_message_images(self.session, [source.db_id], [image_id]))
            ['image_results'][0]['status'], 'selected')
        second = await self.original(2, parts=[MessagePart(PartKind.STICKER, text='🙂'),
            MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=self.red)])
        second_id = (await self.descriptors(second))[second.db_id][0]['image_id']
        retired = await self.store.retire_context_images(self.session, target_images=0)
        self.assertEqual(retired.removed_images, 1)
        _blocks, raw = await self.store.load_live_context(self.session)
        self.assertEqual(raw[0].message.parts[0].text, '🙂')
        self.assertEqual(raw[0].message.parts[-1].text, '[Image compacted]')
        await self.store.reset_context(self.session)
        self.assertEqual(await self.store.list_preview_refs(self.session), set())
        fresh = PostgresStore(self.test_dsn, schema=self.schema)
        try:
            await fresh.initialize()
            opened = await fresh.resolve_message_images(self.session, [source.db_id, second.db_id], [image_id, second_id])
            self.assertEqual([item['status'] for item in opened['image_results']], ['selected', 'selected'])
            self.assertEqual([part.data_b64 for part in opened['evidence_parts'] if part.kind == PartKind.IMAGE],
                [self.red, self.red])
            self.assertEqual((await fresh.read_messages(self.session, [source.db_id]))[0].message, canonical)
            async with fresh.pool.connection() as conn:
                count = (await (await conn.execute('SELECT count(*) AS n FROM message_previews WHERE session_id=%s',
                    (self.session,))).fetchone())['n']
            self.assertEqual(count, 1, 'Identical compressed bytes are stored once per conversation')
            self.assertEqual(await fresh.load_preview_data(self.session, [reference]),
                {reference: base64.b64decode(self.red)})
        finally:
            await fresh.close()

    async def test_source_edit_and_full_reset_deny_old_occurrences_despite_matching_pixels_and_warm_cache(self):
        old = await self.original(1)
        image_id = (await self.descriptors(old))[old.db_id][0]['image_id']
        cache = PreviewCache(self.store)
        try:
            await cache.materialize_many(self.session, [old.message], vision=True)
            old_scope = await self.store.get_scope(self.session)
            revision = await self.original(1, text='Corrected caption, same pixels.')
            with self.assertRaises(StaleScopeError):
                await self.store.describe_message_images(self.session, [old.db_id], expected_scope=old_scope)
            current_id = (await self.descriptors(revision))[revision.db_id][0]['image_id']
            self.assertNotEqual(image_id, current_id)
            self.assertEqual(revision.db_id, old.db_id)
            result = await self.store.resolve_message_images(self.session, [old.db_id], [image_id, current_id])
            self.assertEqual([item['status'] for item in result['image_results']], ['unavailable', 'selected'])
            await self.store.reset_full(self.session, self.config.default_session_settings())
            new = await self.original(1, text='A new agent generation with the same pixels.')
            new_id = (await self.descriptors(new))[new.db_id][0]['image_id']
            result = await self.store.resolve_message_images(self.session, [old.db_id, new.db_id],
                [image_id, current_id, new_id])
            self.assertEqual([item['status'] for item in result['image_results']],
                ['unavailable', 'unavailable', 'selected'])
            self.assertEqual(len(result['evidence_parts']), 2)
            self.assertEqual((await self.descriptors(old, new)).keys(), {new.db_id})
            self.assertEqual(len(await self.store.list_message_revisions(self.session, old.db_id)), 2)
            reference = old.message.parts[-1].preview_ref
            self.assertEqual(await self.store.load_preview_data(self.session, [reference]),
                {reference: base64.b64decode(self.red)}, 'Audit bytes remain although ordinary old-source access is denied')
        finally:
            cache.close()

    async def test_hidden_deleted_foreign_and_unrequested_occurrences_never_gain_access_from_shared_bytes(self):
        hidden = await self.original(1)
        deleted = await self.original(2)
        allowed = await self.original(3)
        foreign = await self.original(1, session='telegram:another-chat')
        described = await self.descriptors(hidden, deleted, allowed)
        foreign_id = (await self.descriptors(foreign, session='telegram:another-chat'))[foreign.db_id][0]['image_id']
        await self.store.hide_message_ids(self.session, [hidden.db_id])
        await self.store.delete_message_ids(self.session, [deleted.db_id])
        ids = [described[hidden.db_id][0]['image_id'], described[deleted.db_id][0]['image_id'],
            foreign_id, described[allowed.db_id][0]['image_id'], 'img:999999:1:0']
        result = await self.store.resolve_message_images(self.session, [hidden.db_id, deleted.db_id, foreign.db_id], ids)
        self.assertTrue(all(item['status'] == 'unavailable' for item in result['image_results']))
        self.assertEqual(result['evidence_parts'], [])
        self.assertTrue(all(set(item) == {'image_id', 'status', 'reason'} for item in result['image_results']))
        self.assertEqual(await self.descriptors(hidden, deleted, foreign), {})
        self.assertTrue((await self.descriptors(allowed))[allowed.db_id][0]['available'])
        self.assertTrue(await self.store.list_message_revisions(self.session, deleted.db_id))

    async def test_lookup_copies_are_not_discovery_seeds_but_independent_original_observations_remain(self):
        source = await self.original(1)
        text = await self.original(2, parts=[MessagePart(PartKind.TEXT, text='Related conversation.')])
        image_id = (await self.descriptors(source))[source.db_id][0]['image_id']
        result = await self.store.resolve_message_images(self.session, [source.db_id], [image_id])
        copied = await self.store.append_message(self.session, ConversationMessage(MessageRole.TOOL,
            result['evidence_parts'], name='memory_read', metadata={'tool_phase': 'result'}))
        synthetic = await self.store.append_message(self.session, ConversationMessage(MessageRole.USER,
            source.message.parts, metadata={'synthetic_role': 'reply_target'}))
        external = await self.store.append_message(self.session, ConversationMessage(MessageRole.TOOL,
            [MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=self.blue)],
            name='remote_screenshot', metadata={'tool_phase': 'result'}))
        described = await self.descriptors(source, text, copied, synthetic, external)
        self.assertEqual(set(described), {source.db_id, text.db_id, external.db_id})
        self.assertEqual(described[text.db_id], [])
        self.assertTrue(described[external.db_id][0]['available'])
        guessed_copy = f'img:{copied.db_id}:1:1'
        selected = await self.store.resolve_message_images(self.session, [copied.db_id], [guessed_copy, image_id])
        self.assertTrue(all(item['status'] == 'unavailable' for item in selected['image_results']))
        self.assertEqual(selected['evidence_parts'], [])

    async def test_stale_scope_is_rejected_before_opening_and_duplicate_selection_does_not_repeat_pixels(self):
        source = await self.original(1)
        image_id = (await self.descriptors(source))[source.db_id][0]['image_id']
        scope = await self.store.get_scope(self.session)
        result = await self.store.resolve_message_images(self.session, [source.db_id], [image_id, image_id], expected_scope=scope)
        self.assertEqual(len(result['image_results']), 1)
        self.assertEqual(len(result['evidence_parts']), 2)
        await self.store.reset_context(self.session)
        with self.assertRaises(StaleScopeError):
            await self.store.describe_message_images(self.session, [source.db_id], expected_scope=scope)
        with self.assertRaises(StaleScopeError):
            await self.store.resolve_message_images(self.session, [source.db_id], [image_id], expected_scope=scope)
        self.assertEqual((await self.store.resolve_message_images(self.session, [source.db_id], [image_id]))
            ['image_results'][0]['status'], 'selected')

    async def test_import_source_lookup_obeys_generation_visibility_and_current_revision(self):
        original = await self.original(1)
        scope = await self.store.get_scope(self.session)
        identity = {'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '1'}
        lookup = self.store.read_message_by_source
        self.assertEqual((await lookup(self.session, **identity, expected_scope=scope)).db_id, original.db_id)
        edited = await self.original(1, text='Corrected source caption.')
        with self.assertRaises(StaleScopeError):
            await lookup(self.session, **identity, expected_scope=scope)
        current = await lookup(self.session, **identity, expected_scope=scope, generation_only=True)
        self.assertEqual(current.message, edited.message)
        await self.store.reset_context(self.session)
        current = await lookup(self.session, **identity, expected_scope=scope, generation_only=True)
        self.assertEqual(current.db_id, original.db_id, 'Import ignores working-context resets, like original append')
        self.assertIsNone(await lookup('telegram:another-chat', **identity))
        hidden = await self.original(2)
        await self.store.hide_message_ids(self.session, [hidden.db_id])
        self.assertIsNone(await lookup(self.session, **{**identity, 'source_message_id': '2'},
            expected_scope=scope, generation_only=True))
        await self.store.reset_full(self.session, self.config.default_session_settings())
        with self.assertRaises(StaleScopeError):
            await lookup(self.session, **identity, expected_scope=scope, generation_only=True)
        self.assertIsNone(await lookup(self.session, **identity))
        fresh = await self.original(1, text='Same Telegram identity in the new generation.')
        self.assertNotEqual(fresh.db_id, original.db_id)
        self.assertEqual((await lookup(self.session, **identity)).db_id, fresh.db_id)
