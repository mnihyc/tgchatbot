from __future__ import annotations

from tgchatbot.config import AppConfig
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.tools.base import ToolSpec
from tgchatbot.tools.file_send import FileSendTool
from tgchatbot.tools.python_exec import PythonExecTool
from tgchatbot.tools.remote_workspace import RemoteWorkspaceClient
from tgchatbot.tools.shell_exec import ShellExecTool
from tgchatbot.tools.sticker_send import StickerQueryTool, StickerSendSelectedTool


class ToolRegistry:
    def __init__(self, config: AppConfig, remote: RemoteWorkspaceClient, sticker_catalog: StickerCatalog) -> None:
        self.config = config
        self.remote = remote
        self.remote_workspace = remote
        self.sticker_catalog = sticker_catalog
        self._python_exec = PythonExecTool(config, remote)
        self._shell_exec = ShellExecTool(config, remote)
        self._file_send = FileSendTool(config, remote)
        self._sticker_query = StickerQueryTool(sticker_catalog)
        self._sticker_send_selected = StickerSendSelectedTool(sticker_catalog)

    def list_tools(self, *, allow_python_exec: bool, allow_stickers: bool) -> list[ToolSpec]:
        tools: list[ToolSpec] = []
        if allow_python_exec and self.remote.enabled:
            tools.extend([self._shell_exec.spec, self._python_exec.spec, self._file_send.spec])
        if allow_stickers and int(self.sticker_catalog.stats().get("stickers", 0)) > 0:
            tools.extend([self._sticker_query.spec, self._sticker_send_selected.spec])
        return tools

    def get(self, name: str) -> ToolSpec | None:
        mapping = {
            self._python_exec.spec.name: self._python_exec.spec,
            self._shell_exec.spec.name: self._shell_exec.spec,
            self._file_send.spec.name: self._file_send.spec,
            self._sticker_query.spec.name: self._sticker_query.spec,
            self._sticker_send_selected.spec.name: self._sticker_send_selected.spec,
        }
        return mapping.get(name)
