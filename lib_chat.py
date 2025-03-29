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

from utils import detect_image

class LChat:
    NAME = ''
    MAX_RCTOKENS = 8192
    MAX_IMAGES = 0
    
    def __init__(self, cid: int, tid: int):
        self.history: List[Dict[str, object]] = []
        self.cid: int = cid
        self.tid: int = tid
        self.a_m_c = None

    async def get_parts(self, message: str|List[str], type: LType) -> List[object]:
        if type == LType.TEXT:
            if isinstance(message, str):
                message = [message]
            return message
        else:
            res = []
            for img in detect_image(message):
                mime, data, _ = img
                res.append({'mime_type': mime, 'data': data})
            return res

    def add_message_callback(self, callback: Callable[[], None]) -> None:
        self.a_m_c = callback
    
    async def add_message(self, parts: List[object], role: LRole) -> None:
        match role:
            case LRole.USER:
                rolestr = 'user'
            case LRole.SYSTEM:
                rolestr = 'system'
            case LRole.MODEL:
                rolestr = 'model'
        self.history.append({'role': rolestr, 'parts': parts})
        if self.a_m_c:
            self.a_m_c()

    async def pop_message(self, index: int = -1) -> Dict[str, object]:
        return self.history.pop(index)

    async def clear_message(self) -> None:
        self.history.clear()
    
    async def rollback_message(self) -> bool:
        await self.revoke_message()
        if len(self.history) > 1:
            self.history.pop()
            return True
        return False

    async def retry_message(self) -> bool:
        if len(self.history) > 1:
            self.history.pop()
            return True
        return False
    
    async def revoke_message(self) -> bool:
        while len(self.history) > 1:
            if self.history[-1]['role'] == 'user':
                self.history.pop()
            else:
                return True
        return False

    async def cut_history(self, max_tokens: int = None, max_images: int = None) -> tuple[int]:
        if max_tokens is None:
            max_tokens = self.MAX_RCTOKENS
        if max_images is None:
            max_images = self.MAX_IMAGES
        cuts = len(self.history)
        imgs, i = 0, len(self.history)
        while i > 0:
            i -= 1
            if self.history[i]['role'] == 'system':
                continue
            if isinstance(self.history[i]['parts'][0], dict):
                imgs += 1
            if imgs > max_images:
                self.history.pop(i)
        while (cur := await self.count_tokens()) > max_tokens:
            for i in range(len(self.history)):
                if self.history[i]['role'] == 'system':
                    continue
                self.history.pop(i)
                break
        return (cuts - len(self.history), cur)
    
    @abstractmethod
    async def count_tokens(self) -> int:
        pass

    @abstractmethod
    async def ask(self) -> str:
        pass

__all__ = ['LRole', 'LType', 'LChat']