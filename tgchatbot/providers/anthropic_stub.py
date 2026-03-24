from __future__ import annotations

from tgchatbot.providers.base import ProviderCapabilities


class AnthropicProvider:
    name = 'anthropic'
    capabilities = ProviderCapabilities(multimodal_input=True, function_tools=True, native_web_search=True)

    async def generate(self, **_: object):
        raise NotImplementedError('Anthropic adapter remains scaffolded in this build.')

    def make_tool_result_items(self, tool_call, tool_output):
        return []
