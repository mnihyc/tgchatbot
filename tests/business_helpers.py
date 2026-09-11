from __future__ import annotations

import copy
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from tgchatbot.config import load_config
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ProviderResponse, ToolResult
from tgchatbot.providers.base import ProviderCapabilities, RequestTokenEstimate
from psycopg import AsyncConnection, sql
from tgchatbot.storage.postgres_store import PostgresStore
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.previews import PreviewCache
from tgchatbot.tools.base import ToolSpec


class ScriptedProvider:
    """Only the external model boundary is replaced; runtime and PostgreSQL are real."""

    capabilities = ProviderCapabilities()

    def __init__(self, name="openai", responses=()):
        self.name = name
        self.responses = list(responses)
        self.requests = []

    async def generate(self, **kwargs):
        request = copy.deepcopy({key: value for key, value in kwargs.items() if key != "tools"})
        # Capture the advertised model contract, not live runners and their
        # PostgreSQL pools/tasks, which are neither serializable nor sent out.
        request["tools"] = [ToolSpec(tool.name, tool.description, copy.deepcopy(tool.parameters_schema), None)
                            for tool in kwargs.get("tools", [])]
        self.requests.append(request)
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
        self.test_dsn = os.getenv("TEST_DATABASE_URL")
        if not self.test_dsn:
            self.skipTest("TEST_DATABASE_URL is required for real PostgreSQL business tests")
        self.schema = f"business_{uuid.uuid4().hex}"
        self._stores = []
        self.addAsyncCleanup(self._cleanup_database)
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
            "DATABASE_URL": self.test_dsn,
        }, clear=True):
            self.config = load_config()
        self.artifact_store = ArtifactStore(self.path / "replay", max_bytes=32 * 1024 * 1024)
        self.preview_cache = PreviewCache(self.path / "previews", max_bytes=32 * 1024 * 1024)
        self.addCleanup(self.preview_cache.close)
        self.store = await self.new_store()
        self.provider = ScriptedProvider()
        self.tools = FixtureTools()
        self.runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools, providers={"openai": self.provider}, preview_cache=self.preview_cache)
        self.session = "telegram:100"

    async def settings(self, **changes):
        settings = await self.store.get_or_create_session(self.session, self.config.default_session_settings())
        for name, value in changes.items():
            setattr(settings, name, value)
        await self.store.save_session(self.session, settings)
        return settings

    async def new_store(self, *, artifact_store=None):
        store = PostgresStore(self.test_dsn, schema=self.schema,
                              artifact_store=artifact_store or self.artifact_store)
        self._stores.append(store)
        await store.initialize()
        return store

    async def _cleanup_database(self):
        for store in self._stores:
            await store.close()
        # Drop only this generated test schema; never use the deployment schema.
        async with await AsyncConnection.connect(self.test_dsn, autocommit=True) as conn:
            await conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(self.schema)))
