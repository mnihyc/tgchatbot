"""Sticker evidence and delivery tests replace only external model/Telegram calls."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from telegram.error import BadRequest

from tests.business_helpers import BusinessTestCase, ScriptedProvider
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import (ChatMode, ConversationMessage, MessagePart, MessageRole,
    OutboundSticker, PartKind, ProcessVisibility, ProviderResponse, StickerMode, StickerTiming, ToolResult)
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.sticker_delivery import StickerDeliveryStore
from tgchatbot.storage.sticker_catalog import StickerCatalogStore
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.tools.base import ToolSpec
from tgchatbot.tools.registry import ToolRegistry
from tgchatbot.transports.sticker_delivery import send_sticker
from tgchatbot.transports.telegram_render import TelegramMessageRenderer
from tgchatbot.transports.telegram_adapter import TelegramBotApp


def candidate_parts(sticker_id, frames=1):
    origin = f'sticker_candidate:{sticker_id}'
    return [MessagePart(kind=PartKind.TEXT,text=f'Candidate {sticker_id}; frames in time order',origin=origin),
        *(MessagePart(kind=PartKind.IMAGE,mime_type='image/png',filename=f'{sticker_id}-{index}.png',
            data_b64=base64.b64encode(f'{sticker_id}-{index}'.encode()).decode(),origin=origin)
          for index in range(frames))]


def function(name, call_id):
    return {'status':'completed','output':[{'type':'function_call','name':name,'call_id':call_id,'arguments':'{}'}]}


class StickerEvidenceWorkflowTests(BusinessTestCase):
    async def test_first_catalog_publication_enables_tools_without_restarting_bot(self):
        await self.settings(mode=ChatMode.ASSIST,sticker_mode=StickerMode.AUTO)
        catalog_store = StickerCatalogStore(self.store)
        await catalog_store.initialize()
        catalog = StickerCatalog(catalog_store,self.path,persona_store=self.store)
        await catalog.aensure_loaded()
        tools = ToolRegistry(self.config,SimpleNamespace(enabled=False),catalog)
        provider = ScriptedProvider(responses=[ProviderResponse(final_text='Before catalog'),ProviderResponse(final_text='After catalog')])
        runtime = AgentRuntime(config=self.config,store=self.store,tool_registry=tools,
            providers={'openai':provider},preview_cache=self.preview_cache)
        await runtime.run_turn(session_id=self.session,user_display_name='Human',incoming_message=ConversationMessage.user_text('Hello'))
        self.assertFalse(any(tool.name=='sticker_query' for tool in provider.requests[-1]['tools']))
        revision = await catalog_store.begin_revision(source_root=str(self.path),recipe={})
        await catalog_store.stage_asset(revision,asset_id='sha256:fixture',content_hash='fixture',
            aliases=[{'path':'fixture.webp','pack':'fixture'}],media={},generated_card={},
            corrections={},card={},provenance={},state='ready')
        await catalog_store.activate(revision)
        await runtime.run_turn(session_id=self.session,user_display_name='Human',incoming_message=ConversationMessage.user_text('Stickers are installed now'))
        self.assertIn('sticker_query',[tool.name for tool in provider.requests[-1]['tools']])
        self.assertIn('sticker_send_selected',[tool.name for tool in provider.requests[-1]['tools']])
        self.assertEqual(catalog.stats()['stickers'],1)

    async def make_provider(self, responses):
        provider = OpenAIResponsesProvider(self.config.openai)
        await provider.aclose()
        self.wire = []
        async def respond(request):
            self.wire.append(json.loads(request.content))
            return httpx.Response(200,json=responses.pop(0))
        provider._client = httpx.AsyncClient(base_url='https://example.invalid/v1/',transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        return provider

    async def test_second_query_retires_old_images_without_native_resurrection_and_refreshes_once(self):
        await self.settings(mode=ChatMode.ASSIST,max_input_images=3,compact_target_images=1,
            compact_trigger_tokens=100000,max_interaction_rounds=3)
        provider = await self.make_provider([function('sticker_query','first'),function('sticker_query','second'),
            {'status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'This expression fits.'}]}]}])
        query = SimpleNamespace(run=AsyncMock(side_effect=[
            ToolResult('','sticker_query',{'ok':True,'candidates':[{'sticker_id':'first'}]},evidence_parts=candidate_parts('first',2)),
            ToolResult('','sticker_query',{'ok':True,'candidates':[{'sticker_id':'second'}]},evidence_parts=candidate_parts('second',2))]))
        self.tools.list_tools.return_value = [ToolSpec('sticker_query','Find expressions',{'type':'object','properties':{}},query)]
        memory = SimpleNamespace(tools=[],fetch_profiles=AsyncMock(return_value={'ok':True,'profiles':[]}))
        runtime = AgentRuntime(config=self.config,store=self.store,tool_registry=self.tools,providers={'openai':provider},
            memory=memory,preview_cache=self.preview_cache)
        old = await runtime.ingest_user_message(session_id=self.session,incoming_message=ConversationMessage(
            role=MessageRole.USER,parts=candidate_parts('old'),metadata={'actor_id':'telegram:user:1','actor_kind':'user'}))
        await runtime.run_turn(session_id=self.session,user_display_name='Human',incoming_message=ConversationMessage.user_text('Show another expression'))
        image_urls = lambda wire: [part['image_url'] for item in wire['input']
            for part in (item.get('output',[]) if isinstance(item.get('output'),list) else item.get('content',[]))
            if isinstance(part,dict) and part.get('type')=='input_image']
        self.assertEqual([len(image_urls(wire)) for wire in self.wire],[1,3,2])
        self.assertTrue(all('c2Vjb25k' in value for value in image_urls(self.wire[-1])))
        final_outputs = [item for item in self.wire[-1]['input'] if item.get('type')=='function_call_output']
        self.assertEqual(len([item for item in final_outputs if item['call_id']=='second']),1)
        self.assertIn('Image compacted',json.dumps(final_outputs))
        memory.fetch_profiles.assert_awaited_once()
        rows = await self.store.list_messages(self.session)
        evidence = [row for row in rows if row.metadata.get('tool_evidence')]
        self.assertEqual([row.metadata['tool_phase'] for row in evidence],['call','result','call','result'])
        self.assertTrue(all(row.role==MessageRole.TOOL for row in evidence))
        self.assertFalse(any(part.data_b64 for row in rows for part in row.parts))
        # A provider switch replays portable TOOL evidence, never a fabricated
        # human sticker message. Retired pictures remain retired.
        gemini = GeminiProvider(replace(self.config.gemini,api_key='mock-key'))
        self.addAsyncCleanup(gemini.aclose)
        settings = await self.settings(provider='gemini',model='gemini-3.8-flash')
        restarted = AgentRuntime(config=self.config,store=self.store,tool_registry=self.tools,providers={'gemini':gemini},preview_cache=self.preview_cache)
        state = await restarted._get_live_state(self.session)
        history = restarted._build_provider_history(state,settings=settings,provider_name='gemini')
        history = await self.preview_cache.materialize_many(self.session, history, vision=True)
        contents = [item for row in history for item in gemini._message_to_contents(row)]
        responses = [part['functionResponse'] for item in contents for part in item.get('parts',[]) if 'functionResponse' in part]
        self.assertEqual(sum(len(response.get('parts',[])) for response in responses),2)
        self.assertEqual([response['id'] for response in responses if response['name']=='sticker_query'],['first','second'])

    async def test_shortlist_admission_keeps_whole_animations_and_reports_reduced_count(self):
        await self.settings(mode=ChatMode.ASSIST,max_input_images=3,compact_target_images=3,
            compact_trigger_tokens=100000,max_interaction_rounds=1)
        provider = await self.make_provider([function('sticker_query','query'),
            {'status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'I found a possibility.'}]}]}])
        runner = SimpleNamespace(run=AsyncMock(return_value=ToolResult('','sticker_query',
            {'ok':True,'candidates':[{'sticker_id':'a'},{'sticker_id':'b'}],'candidate_count':2},
            evidence_parts=candidate_parts('a',2)+candidate_parts('b',2))))
        self.tools.list_tools.return_value=[ToolSpec('sticker_query','Find expressions',{'type':'object'},runner)]
        runtime = AgentRuntime(config=self.config,store=self.store,tool_registry=self.tools,providers={'openai':provider},preview_cache=self.preview_cache)
        result = await runtime.run_turn(session_id=self.session,user_display_name='Human',incoming_message=ConversationMessage.user_text('Only inspect'))
        self.assertEqual(result.stickers,[])
        tool = next(item for item in self.wire[1]['input'] if item.get('type')=='function_call_output')
        output = json.loads(tool['output'][0]['text'])
        self.assertEqual(output['candidate_count'],1)
        self.assertEqual(output['candidates'],[{'sticker_id':'a'}])
        self.assertEqual(sum(part['type']=='input_image' for part in tool['output']),2)
        self.assertIn('reduced',output['evidence_notice'])

    async def test_selected_send_timing_reports_queue_then_actual_acknowledgment(self):
        asset = self.path/'selected.webp'
        asset.write_bytes(b'fixture selected sticker')
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        deliveries = StickerDeliveryStore(self.store)
        await deliveries.initialize()
        for timing in (StickerTiming.SEND_NOW,StickerTiming.AFTER_FINAL):
            with self.subTest(timing=timing):
                await self.settings(mode=ChatMode.ASSIST,max_interaction_rounds=1)
                provider = await self.make_provider([function('sticker_send_selected','selected-'+timing.value),
                    {'status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'A warm reply.'}]}]}])
                sticker = OutboundSticker(asset,source_id=digest,content_sha256=digest,timing=timing)
                runner = SimpleNamespace(run=AsyncMock(return_value=ToolResult('','sticker_send_selected',
                    {'ok':True,'status':'queued','sticker_id':digest,'caption':'A hug','action':'offers an embrace'},stickers=[sticker])))
                self.tools.list_tools.return_value=[ToolSpec('sticker_send_selected','Send chosen asset',{'type':'object'},runner)]
                bot = SimpleNamespace(send_sticker=AsyncMock(return_value=SimpleNamespace(message_id=91)))
                async def emit(event):
                    if event.kind=='sticker':
                        restored = TelegramBotApp._sticker_from_event(event)
                        self.assertEqual(restored.content_sha256,digest)
                        await send_sticker(bot,chat_id=100,sticker=restored,deliveries=deliveries)
                runtime = AgentRuntime(config=self.config,store=self.store,tool_registry=self.tools,
                    providers={'openai':provider},preview_cache=self.preview_cache,sticker_delivery=deliveries)
                result = await runtime.run_turn(session_id=self.session,user_display_name='Human',
                    incoming_message=ConversationMessage.user_text('Send the known one'),emit=emit)
                item = next(item for item in self.wire[-1]['input']
                    if item.get('type')=='function_call_output' and item['call_id']=='selected-'+timing.value)
                output = json.loads(item['output'])
                self.assertEqual(output['delivery_state'],'sent' if timing==StickerTiming.SEND_NOW else 'queued')
                self.assertEqual(output['status'],output['delivery_state'])
                if timing==StickerTiming.AFTER_FINAL:
                    bot.send_sticker.assert_not_awaited()
                    await send_sticker(bot,chat_id=100,sticker=result.stickers[0],deliveries=deliveries)
                bot.send_sticker.assert_awaited_once()
                self.assertEqual((await deliveries.get(sticker.delivery_operation_id))['status'],'sent')
                normalized = runtime._describe_tool_result('sticker_send_selected',{'output':output})
                self.assertIn(digest,normalized)
                self.assertIn('offers an embrace',normalized)

    async def test_sticker_only_reply_sends_selected_asset_without_placeholder_text(self):
        asset = self.path / 'sticker-only.webp'
        asset.write_bytes(b'fixture sticker-only selection')
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        deliveries = StickerDeliveryStore(self.store)
        await deliveries.initialize()
        for timing in (StickerTiming.SEND_NOW, StickerTiming.AFTER_FINAL):
            with self.subTest(timing=timing):
                settings = await self.settings(mode=ChatMode.ASSIST, max_interaction_rounds=1,
                    process_visibility=ProcessVisibility.OFF)
                provider = await self.make_provider([
                    function('sticker_send_selected', 'sticker-only-' + timing.value), {'status': 'completed', 'output': []}])
                sticker = OutboundSticker(asset, source_id=digest, content_sha256=digest, timing=timing)
                runner = SimpleNamespace(run=AsyncMock(return_value=ToolResult('', 'sticker_send_selected',
                    {'ok': True, 'status': 'queued', 'sticker_id': digest}, stickers=[sticker])))
                self.tools.list_tools.return_value = [ToolSpec('sticker_send_selected',
                    'Send chosen asset', {'type': 'object'}, runner)]
                bot = SimpleNamespace(send_sticker=AsyncMock(return_value=SimpleNamespace(message_id=91)),
                    send_message=AsyncMock(return_value=SimpleNamespace(message_id=92)))
                message = SimpleNamespace(chat=SimpleNamespace(id=100), message_id=10, get_bot=lambda: bot)

                async def emit(event):
                    if event.kind == 'sticker':
                        await send_sticker(bot, chat_id=100, sticker=TelegramBotApp._sticker_from_event(event),
                            deliveries=deliveries)

                runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
                    providers={'openai': provider}, preview_cache=self.preview_cache, sticker_delivery=deliveries)
                result = await runtime.run_turn(session_id=self.session, user_display_name='Participant',
                    incoming_message=ConversationMessage.user_text('A comforting sticker only, please.'), emit=emit)
                app = TelegramBotApp.__new__(TelegramBotApp)
                app.config, app.runtime, app.store = self.config, runtime, self.store
                await app._deliver_result(message, SimpleNamespace(), settings, result, sent_before_receipts=[])
                bot.send_sticker.assert_awaited_once()
                self.assertEqual((await deliveries.get(sticker.delivery_operation_id))['status'], 'sent')
                bot.send_message.assert_not_awaited()
                self.assertEqual(result.text, '')

    async def test_explicit_retry_can_choose_a_different_sticker_with_reused_provider_call_id(self):
        await self.settings(mode=ChatMode.ASSIST,max_interaction_rounds=1)
        provider = await self.make_provider([function('sticker_send_selected','reused'),
            {'status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'First choice'}]}]},
            function('sticker_send_selected','reused'),
            {'status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'New choice'}]}]}])
        deliveries = StickerDeliveryStore(self.store)
        await deliveries.initialize()
        selections=[]
        for name in ('first','second'):
            asset=self.path/(name+'.webp'); asset.write_bytes(name.encode())
            digest=hashlib.sha256(asset.read_bytes()).hexdigest()
            selections.append(ToolResult('','sticker_send_selected',{'ok':True,'status':'queued'},
                stickers=[OutboundSticker(asset,source_id=digest,content_sha256=digest,timing=StickerTiming.AFTER_FINAL)]))
        runner=SimpleNamespace(run=AsyncMock(side_effect=selections))
        self.tools.list_tools.return_value=[ToolSpec('sticker_send_selected','Select',{'type':'object'},runner)]
        runtime=AgentRuntime(config=self.config,store=self.store,tool_registry=self.tools,
            providers={'openai':provider},preview_cache=self.preview_cache,sticker_delivery=deliveries)
        trigger=await runtime.ingest_user_message(session_id=self.session,incoming_message=ConversationMessage.user_text('Pick a sticker'))
        first=await runtime.run_turn_from_stored(session_id=self.session,user_display_name='Human',trigger_message_id=trigger.db_id)
        await self.store.hide_messages_since(self.session,trigger.db_id+1)
        runtime.invalidate_session(self.session)
        second=await runtime.run_turn_from_stored(session_id=self.session,user_display_name='Human',trigger_message_id=trigger.db_id)
        self.assertNotEqual(first.stickers[0].delivery_operation_id,second.stickers[0].delivery_operation_id)
        self.assertNotEqual(first.stickers[0].source_id,second.stickers[0].source_id)
        self.assertFalse((await deliveries.begin(first.stickers[0].delivery_operation_id))['may_send'])
        self.assertTrue((await deliveries.begin(second.stickers[0].delivery_operation_id))['may_send'])


class StickerDeliveryWorkflowTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.deliveries = StickerDeliveryStore(self.store)
        await self.deliveries.initialize()
        self.asset = self.path/'original.webp'
        self.asset.write_bytes(b'original sticker bytes')
        self.bot = SimpleNamespace(send_sticker=AsyncMock(return_value=SimpleNamespace(message_id=51)))

    async def queued(self, operation='operation', timing=StickerTiming.AFTER_FINAL):
        scope = await self.store.get_scope(self.session)
        digest = hashlib.sha256(self.asset.read_bytes()).hexdigest()
        await self.deliveries.queue(self.session,'asset',operation_id=operation,expected_scope=scope,
            timing=timing.value, metadata={'content_sha256': digest})
        return OutboundSticker(self.asset,source_id='asset',timing=timing,delivery_operation_id=operation,
            content_sha256=digest)

    async def deliver(self, sticker):
        return await send_sticker(self.bot,chat_id=100,sticker=sticker,deliveries=self.deliveries)

    async def test_confirmed_operation_is_not_sent_twice_and_survives_restart_soft_reset(self):
        sticker = await self.queued()
        first = await self.deliver(sticker)
        second = await self.deliver(sticker)
        self.assertTrue(first['sent'] and second['sent'])
        self.bot.send_sticker.assert_awaited_once()
        self.assertEqual(len(await StickerDeliveryStore(self.store).recent(self.session)),1)
        await self.store.reset_context(self.session)
        self.assertEqual(len(await self.deliveries.recent(self.session)),1)
        await self.store.reset_full(self.session,self.config.default_session_settings())
        self.assertEqual(await self.deliveries.recent(self.session),[])

    async def test_timeout_is_unknown_and_replay_does_not_retry_or_learn_delivery(self):
        sticker = await self.queued()
        self.bot.send_sticker.side_effect=TimeoutError('ack lost')
        first = await self.deliver(sticker)
        await self.deliver(sticker)
        self.assertEqual(first['delivery_state'],'unknown')
        self.bot.send_sticker.assert_awaited_once()
        self.assertEqual(await self.deliveries.recent(self.session),[])
        self.assertEqual(len(await self.deliveries.unresolved(self.session)),1)

    async def test_reset_before_attempt_blocks_send_and_late_ack_stays_old_generation(self):
        old = await self.queued('before')
        await self.store.reset_context(self.session)
        self.assertEqual((await self.deliver(old))['error'],'scope_changed_before_delivery')
        self.bot.send_sticker.assert_not_awaited()
        late = await self.queued('late')
        async def acknowledged_after_reset(**kwargs):
            await self.store.reset_full(self.session,self.config.default_session_settings())
            return SimpleNamespace(message_id=52)
        self.bot.send_sticker.side_effect=acknowledged_after_reset
        self.assertTrue((await self.deliver(late))['sent'])
        self.assertEqual((await self.deliveries.get('late'))['telegram_message_id'],52)
        self.assertEqual(await self.deliveries.recent(self.session),[])

    async def test_changed_asset_and_explicit_rejection_are_failures_without_substitution(self):
        changed = await self.queued('changed')
        self.asset.write_bytes(b'different pixels at same alias')
        self.assertEqual((await self.deliver(changed))['error'],'asset_content_changed')
        self.bot.send_sticker.assert_not_awaited()
        rejected = await self.queued('rejected')
        self.bot.send_sticker.side_effect=BadRequest('invalid sticker')
        self.assertEqual((await self.deliver(rejected))['delivery_state'],'failed')
        self.assertEqual(await self.deliveries.unresolved(self.session),[])

    async def test_canceled_or_crashed_attempt_is_unknown_never_automatically_resent(self):
        canceled = await self.queued('canceled')
        self.bot.send_sticker.side_effect=asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.deliver(canceled)
        self.assertEqual((await self.deliveries.get('canceled'))['status'],'unknown')
        await self.queued('crashed')
        await self.deliveries.begin('crashed')
        restarted = StickerDeliveryStore(self.store)
        await restarted.initialize(recover_interrupted=True)
        self.assertEqual((await restarted.get('crashed'))['status'],'unknown')

    async def test_renderer_uses_same_durable_operation_as_direct_delivery(self):
        sticker = await self.queued()
        source = SimpleNamespace(chat=SimpleNamespace(id=100),message_id=10,get_bot=lambda:self.bot)
        renderer = TelegramMessageRenderer(None,response_delivery=self.config.default_response_delivery,
            min_edit_interval_s=0,source_message=source,sticker_delivery=self.deliveries)
        self.assertTrue((await renderer.send_stickers([sticker]))[0]['sent'])
        self.assertTrue((await self.deliver(sticker))['sent'])
        self.bot.send_sticker.assert_awaited_once()

    async def test_reconstructed_selection_gets_the_actual_prior_receipt(self):
        first = await self.queued()
        await self.deliver(first)
        restored = replace(first, delivery_state='queued', telegram_message_id=None)
        receipt = await self.deliver(restored)
        self.assertEqual(receipt, restored.delivery_receipt())
        self.assertEqual(restored.delivery_state, 'sent')
        self.assertEqual(restored.telegram_message_id, 51)
        self.bot.send_sticker.assert_awaited_once()

    async def test_durable_operation_cannot_be_reused_for_another_selection(self):
        selected = await self.queued()
        for other in (replace(selected, source_id='different-asset'),
                      replace(selected, content_sha256='different-content')):
            with self.assertRaises(ValueError):
                await self.deliver(other)
        self.bot.send_sticker.assert_not_awaited()
        self.assertEqual((await self.deliveries.get('operation'))['status'], 'queued')
        self.assertTrue((await self.deliver(selected))['sent'])

    async def test_queue_retry_preserves_exact_original_timing_and_content(self):
        scope = await self.store.get_scope(self.session)
        await self.deliveries.queue(self.session, 'asset', operation_id='bound', expected_scope=scope,
            timing='after_final', metadata={'content_sha256': 'original'})
        for timing, content in [('send_now', 'original'), ('after_final', 'replacement')]:
            with self.subTest(timing=timing, content=content), self.assertRaises(ValueError):
                await self.deliveries.queue(self.session, 'asset', operation_id='bound', expected_scope=scope,
                    timing=timing, metadata={'content_sha256': content})
        row = await self.deliveries.get('bound')
        self.assertEqual(row['timing'], 'after_final')
        self.assertEqual(row['metadata']['content_sha256'], 'original')

    async def test_concurrent_delivery_attempts_have_one_telegram_owner(self):
        selected = await self.queued()
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed_ack(**kwargs):
            entered.set()
            await release.wait()
            return SimpleNamespace(message_id=61)
        self.bot.send_sticker.side_effect = delayed_ack
        first = asyncio.create_task(self.deliver(selected))
        try:
            await entered.wait()
            replay = replace(selected)
            self.assertEqual((await self.deliver(replay))['delivery_state'], 'sending')
            self.bot.send_sticker.assert_awaited_once()
        finally:
            release.set()
            receipt = await first
        self.assertEqual(receipt['delivery_state'], 'sent')
        self.assertEqual((await self.deliver(replay))['telegram_message_id'], 61)
        self.bot.send_sticker.assert_awaited_once()

    async def test_static_upload_conversion_preserves_pixels_and_original_identity(self):
        import io
        from PIL import Image
        for extension in ('png', 'jpg'):
            with self.subTest(extension=extension):
                self.asset = self.path / f'original.{extension}'
                picture = Image.new('RGBA', (12, 10), (70, 120, 200, 255))
                if extension == 'png':
                    picture.putpixel((0, 0), (20, 30, 40, 0))
                    picture.save(self.asset)
                else:
                    picture.convert('RGB').save(self.asset)
                original = self.asset.read_bytes()
                with Image.open(io.BytesIO(original)) as image:
                    expected = image.convert('RGBA').tobytes()
                selected = await self.queued(extension)
                sent = []
                async def capture(**kwargs):
                    sent.append((kwargs['sticker'].name, kwargs['sticker'].read()))
                    return SimpleNamespace(message_id=71)
                self.bot.send_sticker.side_effect = capture
                receipt = await self.deliver(selected)
                self.assertTrue(receipt['sent'])
                self.assertTrue(sent[0][0].endswith('.webp'))
                with Image.open(io.BytesIO(sent[0][1])) as image:
                    self.assertEqual(image.format, 'WEBP')
                    self.assertEqual(image.convert('RGBA').tobytes(), expected)
                self.assertEqual(self.asset.read_bytes(), original)
                self.assertEqual(selected.content_sha256, hashlib.sha256(original).hexdigest())

    async def test_unsupported_animation_is_not_silently_reduced_to_one_frame(self):
        from PIL import Image
        self.asset = self.path / 'animation.gif'
        first, last = Image.new('RGB', (12, 12), 'red'), Image.new('RGB', (12, 12), 'blue')
        first.save(self.asset, save_all=True, append_images=[last], duration=100, loop=0)
        original = self.asset.read_bytes()
        result = await self.deliver(await self.queued())
        self.assertFalse(result['sent'])
        self.assertEqual(result['error'], 'unsupported_sticker_animation')
        self.bot.send_sticker.assert_not_awaited()
        self.assertEqual(self.asset.read_bytes(), original)

    async def test_verified_original_is_frozen_before_the_telegram_request(self):
        original = self.asset.read_bytes()
        selected = await self.queued()
        sent = []
        async def replace_source_during_send(**kwargs):
            self.asset.write_bytes(b'replacement after hash verification')
            sent.append(kwargs['sticker'].read())
            return SimpleNamespace(message_id=72)
        self.bot.send_sticker.side_effect = replace_source_during_send
        result = await self.deliver(selected)
        self.assertTrue(result['sent'])
        self.assertEqual(sent, [original])

    async def test_each_delivery_preserves_its_source_topic(self):
        for field, value in (('message_thread_id', 37), ('direct_messages_topic_id', 17)):
            with self.subTest(field=field):
                selected = await self.queued(field)
                result = await send_sticker(self.bot, chat_id=100, sticker=selected,
                    deliveries=self.deliveries, **{field: value})
                self.assertTrue(result['sent'])
                request = self.bot.send_sticker.await_args.kwargs
                self.assertEqual(request[field], value)
                other = 'message_thread_id' if field == 'direct_messages_topic_id' else 'direct_messages_topic_id'
                self.assertNotIn(other, request)
