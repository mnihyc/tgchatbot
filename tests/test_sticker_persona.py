"""A catalog handle must observe the same persona and deliveries as a fresh one."""
from tests.business_helpers import BusinessTestCase
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.stickers.plan import StickerRetrievalPlan
from tgchatbot.storage.sticker_delivery import StickerDeliveryStore


class StickerPersonaWorkflowTests(BusinessTestCase):
    async def test_existing_handle_observes_preference_changes_and_full_reset(self):
        await self.settings()
        deliveries = StickerDeliveryStore(self.store)
        await deliveries.initialize()
        existing = StickerCatalog(None, self.path, persona_store=self.store, delivery_store=deliveries)
        familiar = {'visual_identity': {'character_archetype': 'cat'}}
        changed = {'visual_identity': {'character_archetype': 'bird'}}
        await self.store.save_sticker_persona(self.session, familiar)
        self.assertEqual((await existing.adescribe_persona_context(self.session))['effective_persona'], familiar)
        scope = await self.store.get_scope(self.session)
        await deliveries.queue(self.session, 'chosen', operation_id='one', expected_scope=scope,
                               timing='after_final')
        await deliveries.begin('one')
        await deliveries.finish('one', 'sent', telegram_message_id=1)
        self.assertEqual((await existing.adescribe_style_context(self.session))['recent_sticker_ids'], ['chosen'])

        await self.store.save_sticker_persona(self.session, changed)
        restarted = StickerCatalog(None, self.path, persona_store=self.store, delivery_store=deliveries)
        self.assertEqual(await existing.adescribe_persona_context(self.session),
                         await restarted.adescribe_persona_context(self.session))
        await self.store.reset_context(self.session)
        self.assertEqual((await existing.adescribe_persona_context(self.session))['effective_persona'], changed)
        self.assertEqual((await existing.adescribe_style_context(self.session))['recent_sticker_ids'], ['chosen'])

        await self.store.reset_full(self.session, self.config.default_session_settings())
        self.assertEqual((await existing.adescribe_persona_context(self.session))['effective_persona'], {})
        self.assertEqual(await existing.adescribe_persona_context(self.session),
                         await restarted.adescribe_persona_context(self.session))
        self.assertEqual((await existing.adescribe_style_context(self.session))['recent_sticker_ids'], [])
        # An explicit remember operation after reset must not merge retired identity.
        plan = StickerRetrievalPlan.from_payload({'intent_core': 'A greeting',
            'persona_mode': 'merge_and_remember', 'persona': {'affect_profile': {'default_tone': 'warm'}}})
        await existing.aprepare_query_context(plan=plan, session_id=self.session, persist_persona=True,
            expected_scope=await self.store.get_scope(self.session))
        self.assertEqual(await self.store.get_sticker_persona(self.session),
                         {'affect_profile': {'default_tone': 'warm'}})
