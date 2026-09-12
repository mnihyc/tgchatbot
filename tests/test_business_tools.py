from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import OutboundArtifact
from tgchatbot.tools.base import ToolContext
from tgchatbot.tools.file_send import FileSendTool
from tgchatbot.tools.python_exec import PythonExecTool
from tgchatbot.tools.registry import ToolRegistry
from tgchatbot.tools.shell_exec import ShellExecTool


class ToolWorkflowTests(BusinessTestCase):
    async def test_file_send_fetches_requested_paths_without_implicit_execution(self):
        artifact = OutboundArtifact(self.path / "report.txt", "report.txt")
        remote = SimpleNamespace(fetch_files=AsyncMock(return_value=[artifact]))
        tool = FileSendTool(self.config, remote)
        result = await tool.run({"scope": "outputs", "paths": [" report.txt ", "report.txt", ""]}, ToolContext(self.session, "tester"))
        remote.fetch_files.assert_awaited_once_with(session_id=self.session, remote_paths=["report.txt"], scope="outputs")
        self.assertTrue(result.output["ok"])
        self.assertEqual(result.artifacts, [artifact])
        self.assertEqual(result.output["prepared_files"], ["report.txt"])
        self.assertEqual(result.output["delivery_state"], "pending")

    async def test_file_send_validation_and_empty_result_are_model_visible_errors(self):
        remote = SimpleNamespace(fetch_files=AsyncMock(return_value=[]))
        tool = FileSendTool(self.config, remote)
        for args in ({"scope": "outside"}, {"paths": "not-an-array"},
                     {'scope': 'outputs', 'paths': []}, {'paths': ['  ']}, {}):
            with self.subTest(args=args):
                result = await tool.run(args, ToolContext(self.session, "tester"))
                self.assertFalse(result.output["ok"])
        remote.fetch_files.assert_not_awaited()
        result = await tool.run({"scope": "outputs", "paths": ['missing.txt']}, ToolContext(self.session, "tester"))
        self.assertFalse(result.output["ok"])
        self.assertEqual(result.artifacts, [])

    async def test_remote_execution_does_not_automatically_send_files(self):
        remote_result = {"ok": True, "returncode": 0, "stdout": "report ready", "stderr": ""}
        remote = SimpleNamespace(run_shell=AsyncMock(return_value=remote_result), run_python=AsyncMock(return_value=remote_result))
        cases = [(ShellExecTool(self.config, remote), {"command": "mock", "timeout_s": 10**6}, remote.run_shell), (PythonExecTool(self.config, remote), {"code": "mock", "timeout_s": 10**6}, remote.run_python)]
        for tool, args, run in cases:
            with self.subTest(tool=tool.spec.name):
                result = await tool.run(args, ToolContext(self.session, "tester"))
                self.assertTrue(result.output["ok"])
                self.assertEqual(result.artifacts, [])
                self.assertEqual(run.await_args.kwargs["timeout_s"], self.config.ssh_exec.max_tool_timeout_s)
                self.assertEqual(run.await_args.kwargs["session_id"], self.session)

    async def test_registry_requires_remote_enabled_and_sticker_catalog(self):
        remote = SimpleNamespace(enabled=False)
        catalog = Mock()
        catalog.stats.return_value = {"stickers": 0}
        registry = ToolRegistry(self.config, remote, catalog)
        self.assertEqual(registry.list_tools(allow_python_exec=True, allow_stickers=True), [])
        remote.enabled = True
        names = {tool.name for tool in registry.list_tools(allow_python_exec=True, allow_stickers=False)}
        self.assertEqual(names, {"shell_exec", "python_exec", "file_send"})
        catalog.stats.return_value = {"stickers": 1}
        names = {tool.name for tool in registry.list_tools(allow_python_exec=False, allow_stickers=True)}
        self.assertEqual(names, {"sticker_query", "sticker_send_selected"})
