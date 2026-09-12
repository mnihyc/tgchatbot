"""UTC originals and one configured presentation zone survive live turns and compaction."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb
from telegram import Chat, Message, Update, User

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.compaction_schema import compaction_json_schema
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ChatMode, ProviderResponse
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.postgres_store import PostgresStore
from tgchatbot.storage.previews import PreviewCache
from tgchatbot.transports.telegram_adapter import TelegramBotApp


class RuntimeTimezoneTests(BusinessTestCase):
    def intake_app(self):
        # Bypass only construction of a network-connected Telegram application.
        app = TelegramBotApp.__new__(TelegramBotApp)
        app.config, app.runtime, app.store = self.config, self.runtime, self.store
        app.artifact_store = ArtifactStore(self.config.artifact_dir)
        app.remote_workspace = SimpleNamespace(enabled=False)
        app._chat_states = {}
        app._ensure_reply_worker = AsyncMock()
        app._notify_user_error = AsyncMock()
        return app

    async def live_message(self, app, *, source_id, instant, text):
        message = Message(message_id=source_id, date=instant,
            chat=Chat(id=100, type='private'), from_user=User(id=7, first_name='Alex', is_bot=False),
            text=text)
        update = Update(update_id=source_id, message=message)
        await app._ingest_update(update, should_reply=False, is_group=False, is_edit=False)
        app._notify_user_error.assert_not_awaited()
        originals = await self.store.list_canonical_messages(self.session)
        return next(row for row in originals if row.message.metadata.get('source_message_id') == str(source_id))

    async def assert_cold_parity(self, settings):
        live = await self.runtime._get_live_state(self.session)
        reader = await self.new_store()
        cache = PreviewCache(reader, max_bytes=0)
        self.addCleanup(cache.close)
        cold = AgentRuntime(config=self.config, store=reader, tool_registry=self.tools,
            providers=self.runtime.providers, preview_cache=cache)
        rebuilt = await cold._get_live_state(self.session)
        self.assertEqual(live.raw_messages, rebuilt.raw_messages)
        self.assertEqual(live.blocks, rebuilt.blocks)
        warm_history = self.runtime._build_provider_history(live,
            settings=settings, provider_name=settings.provider)
        cold_history = cold._build_provider_history(rebuilt,
            settings=settings, provider_name=settings.provider)
        self.assertEqual(warm_history, cold_history)
        return warm_history

    async def test_live_utc_originals_use_local_provider_wire_and_reconstruct_with_non_utc_database_default(self):
        # A server/connection default must not change the store's UTC contract.
        dsn = make_conninfo(self.test_dsn, options='-c timezone=America/New_York')
        store = PostgresStore(dsn, schema=self.schema)
        self._stores.append(store)
        await store.initialize()
        self.store = store
        self.runtime.store = store
        self.runtime.preview_cache = PreviewCache(store, max_bytes=0)
        self.addCleanup(self.runtime.preview_cache.close)
        self.config = replace(self.config, default_metadata_timezone='Asia/Singapore')
        self.runtime.config = self.config
        async with store.pool.connection() as conn:
            self.assertEqual((await (await conn.execute('SHOW TimeZone')).fetchone())['TimeZone'], 'UTC')

        wire = []
        def respond(request):
            wire.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                'content': {'role': 'model', 'parts': [{'text': 'It was September 7 here.'}]}}]})

        provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        await provider.aclose()
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers['gemini'] = provider
        settings = await self.settings(provider='gemini', model='gemini-3.8-flash', mode=ChatMode.ASSIST)
        app = self.intake_app()
        instant = datetime(2026, 9, 6, 16, 24, 8, tzinfo=timezone.utc)
        literal = 'The literal note says 2026-09-06T16:24:08+00:00.'
        original = await self.live_message(app, source_id=1, instant=instant, text=literal)
        self.assertEqual(original.created_at, int(instant.timestamp()))
        async with store.pool.connection() as conn:
            row = await (await conn.execute('SELECT sent_at FROM messages WHERE id=%s',
                (original.db_id,))).fetchone()
            self.assertEqual(row['sent_at'].utcoffset(), timedelta(0))
        self.assertEqual(original.message.metadata['sent_at'], instant.isoformat())
        self.assertIn(instant.isoformat(), original.message.parts[0].text)

        await self.runtime.run_turn_from_stored(session_id=self.session,
            user_display_name='Alex', trigger_message_id=original.db_id)
        self.assertEqual(len(wire), 1)
        sent_texts = [part['text'] for content in wire[0]['contents']
            for part in content['parts'] if 'text' in part]
        provenance = next(text for text in sent_texts if text.startswith('[Message provenance:'))
        reply_target = next(text for text in sent_texts if text.startswith('[Application reply target:'))
        self.assertIn('2026-09-07T00:24:08+08:00', provenance)
        self.assertIn('2026-09-07T00:24:08+08:00', reply_target)
        self.assertNotIn(instant.isoformat(), provenance)
        self.assertNotIn(instant.isoformat(), reply_target)
        self.assertIn(literal, sent_texts, 'Timestamp-looking original prose belongs to the sender.')
        self.assertEqual((await store.read_messages(self.session, [original.db_id]))[0], original)
        await self.assert_cold_parity(settings)

    async def test_environment_zone_handles_dst_and_legacy_per_chat_controls_cannot_override_it(self):
        self.config = replace(self.config, default_metadata_timezone='America/New_York')
        self.runtime.config = self.config
        await self.settings(mode=ChatMode.ASSIST)
        async with self.store.pool.connection() as conn:
            await conn.execute('UPDATE sessions SET settings=settings || %s WHERE session_id=%s',
                (Jsonb({'metadata_timezone': 'Asia/Singapore'}), self.session))
        app = self.intake_app()
        reply = AsyncMock()
        command = SimpleNamespace(effective_chat=Chat(id=100, type='private'),
            effective_user=User(id=7, first_name='Alex', is_bot=False),
            effective_message=SimpleNamespace(reply_text=reply))
        before = await self.store.get_or_create_session(self.session, self.config.default_session_settings())
        await app.param_command(command, SimpleNamespace(args=['metadata_timezone', 'UTC']))
        self.assertIn('Unknown parameter', reply.await_args.args[0])
        settings = await self.store.get_or_create_session(self.session, self.config.default_session_settings())
        self.assertEqual(asdict(settings), asdict(before))
        self.assertEqual(self.config.default_metadata_timezone, 'America/New_York')

        for source_id, instant, shown in (
            (1, datetime(2026, 1, 1, 2, tzinfo=timezone.utc), '2025-12-31T21:00:00-05:00'),
            (2, datetime(2026, 7, 1, 2, tzinfo=timezone.utc), '2026-06-30T22:00:00-04:00'),
        ):
            with self.subTest(instant=instant):
                original = await self.live_message(app, source_id=source_id, instant=instant,
                    text=f'Identify the local date for message {source_id}.')
                self.provider.responses = [ProviderResponse(final_text='Received.')]
                await self.runtime.run_turn_from_stored(session_id=self.session,
                    user_display_name='Alex', trigger_message_id=original.db_id)
                messages = self.provider.requests[-1]['messages']
                presented = next(message for message in messages
                    if message.metadata.get('source_message_id') == str(source_id))
                self.assertIn(shown, presented.parts[0].text)
                targets = [message for message in messages if message.metadata.get('synthetic_role') == 'reply_target']
                self.assertIn(shown, '\n'.join(part.text or '' for part in targets[-1].parts))
                self.assertEqual(original.message.metadata['sent_at'], instant.isoformat())
        await self.assert_cold_parity(settings)

    async def test_compaction_presents_local_source_dates_and_persists_utc_bounds(self):
        self.config = replace(self.config, default_metadata_timezone='Asia/Singapore')
        self.runtime.config = self.config
        # Keep one newest original raw while the preceding conversation forms an episode.
        settings = await self.settings(min_raw_messages_reserve=1)
        app = self.intake_app()
        originals = []
        literal_span = '- Time span: 2026-09-06T16:24:08+00:00 .. 2026-09-06T16:25:08+00:00'
        for source_id, minute, text in (
            (1, 24, 'Bring the blue ticket. The note literally reads:\n' + literal_span),
            (2, 25, 'I will bring that ticket.'),
            (3, 26, 'The latest question stays raw.'),
        ):
            originals.append(await self.live_message(app, source_id=source_id,
                instant=datetime(2026, 9, 6, 16, minute, 8, tzinfo=timezone.utc), text=text))
        candidate = {key: [] for key in compaction_json_schema('episode')['properties']}
        candidate.update(scope='A ticket commitment.\n' + literal_span, interaction_mode='chat_or_sharing',
            decisions=['Participant will bring the blue ticket.'],
            retained_raw_excerpts=['The note literally reads:\n' + literal_span])
        self.provider.responses = [ProviderResponse(final_text=json.dumps(candidate))]
        state = await self.runtime._get_live_state(self.session)
        self.assertTrue(await self.runtime._compact_old_context(session_id=self.session,
            settings=settings, provider=self.provider, state=state, pressure=True))
        self.assertEqual(len(self.provider.requests), 1)
        source_texts = [part.text for message in self.provider.requests[0]['messages']
            for part in message.parts if part.text]
        self.assertIn('2026-09-07T00:24:08+08:00', '\n'.join(source_texts))
        self.assertIn('2026-09-07T00:25:08+08:00', '\n'.join(source_texts))
        metadata_prompt = next(message for message in self.provider.requests[0]['messages']
            if message.metadata.get('source_role') == 'compaction_metadata')
        self.assertNotIn('2026-09-06T16:24:08+00:00', metadata_prompt.parts[0].text)
        evidence = next(json.loads(message.parts[0].text) for message in self.provider.requests[0]['messages']
            if message.metadata.get('message_id') == originals[0].db_id)
        self.assertEqual(evidence['sent_at'], '2026-09-07T00:24:08+08:00')
        self.assertIn(literal_span, '\n'.join(fragment['text'] for fragment in evidence['fragments']))
        blocks = await self.store.list_memory_blocks(self.session)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].kind, 'episode')
        self.assertEqual(blocks[0].time_start, '2026-09-06T16:24:08+00:00')
        self.assertEqual(blocks[0].time_end, '2026-09-06T16:25:08+00:00')
        self.assertEqual(state.raw_messages[-1].db_id, originals[-1].db_id)
        self.assertEqual(await self.store.read_messages(self.session, [row.db_id for row in originals]), originals)
        history = await self.assert_cold_parity(settings)
        summary = next(message for message in history if message.parts[0].text.startswith('[Memory episode'))
        self.assertIn('2026-09-07T00:24:08+08:00', summary.parts[0].text)
        self.assertIn('## Scope\n- ' + candidate['scope']
            + '\n- Time span: 2026-09-07T00:24:08+08:00 .. 2026-09-07T00:25:08+08:00', summary.parts[0].text)
        self.assertEqual(summary.parts[0].text.count(literal_span), 2,
            'The multiline scope and retained quote are model-owned text, not generated date headers.')
