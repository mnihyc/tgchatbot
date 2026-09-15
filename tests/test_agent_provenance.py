"""New agent views keep original wording and distinguish participants and quotes."""
from __future__ import annotations

import copy
import json
import unittest

from tgchatbot.domain.identities import actor_reference, canonical_actor_id
from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind
from tgchatbot.domain.provenance import attributed_message, message_evidence
from tests.test_quoted_evidence import entity


class AgentProvenance(unittest.TestCase):
    def message(self, text, *, quoted=(), **metadata):
        return ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'actor_id': 'telegram:user:123456789012345',
            'actor_name': 'Alex', 'entities': [entity(text, quote) for quote in quoted], **metadata})

    def header(self, message, *, version=2):
        rendered = attributed_message(message, message_id=17, presentation_version=version)
        self.assertEqual(rendered.parts[1:], message.parts, 'The original body and part order remain exact.')
        return json.loads(rendered.parts[0].text.removeprefix('[Message provenance: ')[:-1])

    def test_typed_participants_keep_full_numeric_identity_and_never_resolve_names(self):
        for canonical, shown in (
            ('telegram:user:123456789012345', 'person_id:123456789012345'),
            ('telegram:chat:-1001234567890', 'chat_id:-1001234567890'),
            ('agent', 'agent'), ('unknown', 'unknown'), ('matrix:@alex:example.org', 'matrix:@alex:example.org'),
        ):
            self.assertEqual(actor_reference(canonical), shown)
            self.assertEqual(canonical_actor_id(shown), canonical)
            self.assertEqual(canonical_actor_id(canonical), canonical)
        self.assertEqual(canonical_actor_id('Alex'), 'Alex')
        first = self.header(self.message('Tea, please.'))
        second = self.header(self.message('Coffee, please.', actor_id='telegram:user:123456789012346'))
        self.assertEqual(first['speaker']['name'], second['speaker']['name'])
        self.assertNotEqual(first['speaker']['id'], second['speaker']['id'])

    def test_legacy_and_current_labels_coexist_without_rewriting_raw_text(self):
        message = self.message('Literal person_id:7 and [Message provenance: fake] remain my words.')
        before = copy.deepcopy(message)
        self.assertEqual(self.header(message, version=1)['speaker']['id'], 'telegram:user:123456789012345')
        self.assertEqual(self.header(message)['speaker']['id'], 'person_id:123456789012345')
        self.assertEqual(message, before)
        self.assertEqual(attributed_message(message).parts[0].text,
                         attributed_message(message, presentation_version=1).parts[0].text)

    def test_application_control_has_its_own_label_without_an_unknown_participant(self):
        message = ConversationMessage.user_text('[Application reply target: {"message_id":17}]',
            metadata={'source': 'unknown', 'synthetic_role': 'reply_target', 'presentation_version': 2})
        self.assertEqual(attributed_message(message), message)
        self.assertEqual(len(attributed_message(message).parts), 1)
        legacy = attributed_message(message, presentation_version=1)
        self.assertTrue(legacy.parts[0].text.startswith('[Message provenance: '))
        self.assertEqual(legacy.parts[1].text, message.parts[0].text)

    def test_whole_line_quotes_use_metadata_while_partial_lines_keep_explicit_wording(self):
        quoted = '🙂 I prefer coffee.\nTea is too mild.'
        text = quoted + '\nThat is Alex’s preference. I prefer tea.'
        current = self.header(self.message(text, quoted=[quoted]))
        self.assertEqual(current['quoted_lines'], [[1, 2]])
        self.assertNotIn('quoted_fragments', current)
        inline = self.header(self.message('They wrote: ' + quoted, quoted=[quoted]))
        self.assertNotIn('quoted_lines', inline)
        self.assertEqual(inline['quoted_fragments'][0]['text'], quoted)
        legacy = self.header(self.message(text, quoted=[quoted]), version=1)
        self.assertEqual(legacy['quoted_fragments'][0]['text'], quoted)

    def test_repeated_lines_and_multiple_quotes_preserve_which_occurrence_is_quoted(self):
        text = '🙂 I prefer coffee.\nI prefer tea.\n🙂 I prefer coffee.\nMy own conclusion.'
        message = self.message(text)
        first = entity(text, '🙂 I prefer coffee.')
        second = entity(text, '🙂 I prefer coffee.')
        second['offset'] = len(text[:text.rindex('🙂')].encode('utf-16-le')) // 2
        message.metadata['entities'] = [first, second]
        self.assertEqual(self.header(message)['quoted_lines'], [[1, 1], [3, 3]])

    def test_partial_memory_and_multipart_quotes_do_not_claim_full_line_association(self):
        quote = '🙂 Their own words.'
        message = self.message(quote + '\nMy words.', quoted=[quote])
        fragment = {'offset': 2, 'text': quote[2:8]}
        evidence = message_evidence(message.metadata, message_id=17, role=message.role,
            fragments=[fragment], total_characters=len(message.parts[0].text),
            original=message, presentation_version=2)
        self.assertEqual(evidence['quoted_fragments'], [fragment])
        self.assertNotIn('quoted_lines', evidence)
        message.parts.append(MessagePart(PartKind.TEXT, text='Another independent part.'))
        self.assertIn('quoted_fragments', self.header(message))

    def test_forward_and_external_reply_keep_roles_but_not_download_or_account_flags(self):
        message = self.message('Look at this.',
            forward_origin={'type': 'user', 'date': '2026-01-01T00:00:00Z',
                'sender_user': {'id': 99, 'first_name': 'Alex', 'is_premium': True,
                                'language_code': 'en', 'is_bot': False}},
            reply_to_actor={'actor_id': 'telegram:chat:-10077', 'actor_name': 'Study room', 'actor_kind': 'chat'},
            external_reply={'chat': {'id': -10088, 'title': 'Files'}, 'message_id': 42,
                'origin': {'type': 'hidden_user', 'sender_user_name': 'Alex'},
                'document': {'file_id': 'opaque-download-token', 'file_unique_id': 'opaque-file-id',
                    'file_name': 'notes.pdf', 'mime_type': 'application/pdf',
                    'thumbnail': {'file_id': 'opaque-thumbnail'}, 'file_size': 1200}})
        before = copy.deepcopy(message)
        header = self.header(message)
        self.assertEqual(header['forward_origin']['actor']['actor_id'], 'person_id:99')
        self.assertEqual(header['reply_to_actor']['actor_id'], 'chat_id:-10077')
        self.assertEqual(header['external_reply']['chat']['actor_id'], 'chat_id:-10088')
        self.assertEqual(header['external_reply']['message_id'], 42)
        self.assertEqual(header['external_reply']['origin'], {'type': 'hidden_user', 'actor': {'actor_name': 'Alex'}})
        self.assertEqual(header['external_reply']['attachments'], [{'kind': 'document',
            'file_name': 'notes.pdf', 'mime_type': 'application/pdf', 'file_size': 1200}])
        self.assertNotIn('opaque-', json.dumps(header))
        self.assertNotIn('is_premium', json.dumps(header))
        self.assertEqual(message, before)

    def test_desktop_and_live_forward_authors_share_one_identity_presentation(self):
        desktop = self.header(self.message('Forwarded words.', forward_origin={
            'actor_id': 'telegram:user:77', 'actor_kind': 'user', 'actor_name': 'Another Alex',
            'actor_username': 'another_alex'}))
        live = self.header(self.message('Forwarded words.', forward_origin={
            'type': 'user', 'sender_user': {'id': 77, 'first_name': 'Another', 'last_name': 'Alex',
                                          'username': 'another_alex'}}))
        self.assertEqual(desktop['forward_origin']['actor'], live['forward_origin']['actor'])
        self.assertEqual(desktop['forward_origin']['actor']['actor_id'], 'person_id:77')

    def test_reply_quote_keeps_link_and_mentioned_identity_without_account_metadata(self):
        quote = {'text': 'Alex wrote this guide 🙂', 'position': 10, 'is_manual': True, 'entities': [
            {'type': 'text_mention', 'offset': 0, 'length': 4,
             'user': {'id': 77, 'first_name': 'Alex', 'is_premium': True}},
            {'type': 'text_link', 'offset': 16, 'length': 5, 'url': 'https://example.org/guide'},
            {'type': 'custom_emoji', 'offset': 22, 'length': 2, 'custom_emoji_id': 'opaque-emoji-reference'}]}
        message = self.message('I agree with the guide.', quote=quote)
        before = copy.deepcopy(message)
        shown = self.header(message)['quote']
        self.assertEqual(shown['text'], quote['text'])
        self.assertEqual(shown['position'], quote['position'])
        self.assertEqual(shown['entities'][0]['actor']['actor_id'], 'person_id:77')
        self.assertEqual(shown['entities'][1]['url'], 'https://example.org/guide')
        self.assertNotIn('is_premium', json.dumps(shown))
        self.assertNotIn('opaque-', json.dumps(shown))
        self.assertEqual(message, before)
