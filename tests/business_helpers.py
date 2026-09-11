from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from tgchatbot.config import load_config
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ProviderResponse, ToolResult
from tgchatbot.providers.base import ProviderCapabilities, RequestTokenEstimate
from tgchatbot.storage.sqlite_store import SQLiteStore
from tgchatbot.tools.base import ToolSpec


class ScriptedProvider:
    """Only the external model boundary is replaced; runtime and SQLite are real."""

    capabilities = ProviderCapabilities()

    def __init__(self, name="openai", responses=()):
        self.name = name
        self.responses = list(responses)
        self.requests = []

    async def generate(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("Unexpected model request: extend the explicit test script")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def estimate_request_tokens(self, **kwargs):
        history = kwargs.get("history_tokens_override")
        if history is None:
            history = sum(TokenEstimator.estimate_message(m) for m in kwargs["messages"])
        return RequestTokenEstimate.compose(
            history_tokens=history,
            instructions_tokens=TokenEstimator.estimate_text(kwargs["instructions"]),
            tools_tokens=20 * len(kwargs["tools"]),
        )

    def persistent_history_items(self, response):
        return response.continuation_items

    def make_tool_result_items(self, call, output):
        return [{"type": "function_call_output", "call_id": call.call_id, "output": output}]

    def describe_controls(self, settings):
        return {}


class FixtureTools:
    def __init__(self):
        self.runner = SimpleNamespace(run=AsyncMock(return_value=ToolResult("", "shell_exec", {"ok": True})))
        self.spec = ToolSpec("shell_exec", "Mock remote shell", {"type": "object", "properties": {}}, self.runner)
        self.list_tools = Mock(return_value=[self.spec])
        self.get = Mock(side_effect=lambda name: self.spec if name == self.spec.name else None)


class BusinessTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Keep fixtures inside the checkout, isolated from deployment ./data.
        self.temp = tempfile.TemporaryDirectory(prefix="fixture-", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        with patch.dict(os.environ, {
            "APP_DATA_DIR": str(self.path),
            "TGBOT_TOKEN": "123456:mock-token",
            "DEFAULT_PROVIDER": "openai",
            "OPENAI_API_KEY": "mock-openai-key",
            "DEFAULT_PROVIDER_RETRY_COUNT": "0",
        }, clear=True):
            self.config = load_config()
        self.store = SQLiteStore(self.config.db_path)
        self.provider = ScriptedProvider()
        self.tools = FixtureTools()
        self.runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools, providers={"openai": self.provider})
        self.session = "telegram:100"

    async def settings(self, **changes):
        settings = await self.store.get_or_create_session(self.session, self.config.default_session_settings())
        for name, value in changes.items():
            setattr(settings, name, value)
        await self.store.save_session(self.session, settings)
        return settings
