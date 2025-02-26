from enum import Enum
class LRole(Enum):
    USER = 0
    MODEL = 1
    SYSTEM = 2

class LType(Enum):
    TEXT = 0
    IMAGE = 1

from typing import Callable, Dict, List
from abc import abstractmethod

class LChat:
    NAME = ''
    MAX_RCTOKENS = 1048576 // 2
    MAX_IMAGES = 3000 // 2
    
    def __init__(self, cid: int, tid: int):
        self.history: List[Dict[str, object]] = []
        self.cid: int = cid
        self.tid: int = tid
        self.a_m_c = None

    @abstractmethod
    async def add_message(self, parts: List[object], role: LRole) -> None:
        pass

    def add_message_callback(self, callback: Callable[[], None]) -> None:
        self.a_m_c = callback

    @abstractmethod
    async def get_parts(self, message: str|List[str], type: LType) -> List[object]:
        pass

    async def pop_message(self, index: int = -1) -> Dict[str, object]:
        return self.history.pop(index)

    async def clear_message(self) -> None:
        self.history.clear()
    
    async def rollback_message(self) -> bool:
        if len(self.history) > 1:
            self.history.pop()
            if len(self.history) > 1:
                self.history.pop()
            return True
        return False

    async def retry_message(self) -> bool:
        if len(self.history) > 1:
            self.history.pop()
            return True
        return False

    @abstractmethod
    async def cut_history(self, max_tokens: int = MAX_RCTOKENS, max_images: int = MAX_IMAGES) -> None:
        pass
    
    @abstractmethod
    async def count_tokens(self) -> int:
        pass

    @abstractmethod
    async def ask(self) -> str:
        pass

__all__ = ['LRole', 'LType', 'LChat']