from enum import Enum
class Role(Enum):
    USER = 0
    MODEL = 1
    SYSTEM = 2

from typing import Dict, List

class Chat:
    NAME = ''
    
    def __init__(self, cid: int):
        self.history: List[Dict[str, str]] = []
        self.cid: int = cid

    async def add_message(self, message: str, role: Role) -> None:
        pass

    async def pop_message(self, index: int = -1) -> Dict[str, str]:
        return self.history.pop(index)

    async def clear_message(self) -> None:
        self.history.clear()
    
    async def rollback_message(self) -> bool:
        if len(self.history) > 2:
            self.history.pop()
            self.history.pop()
            return True
        return False
    
    async def count_tokens(self) -> int:
        pass

    async def ask(self) -> str:
        pass

__all__ = ['Role', 'Chat']