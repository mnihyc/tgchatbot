from dotenv import load_dotenv
load_dotenv()

import os
API_KEY = os.getenv('GEMINI_API_KEY')
assert API_KEY, 'GEMINI_API_KEY not set'
API_KEY = [k.strip() for k in API_KEY.split() if k.strip()]

from typing import AsyncIterator, Dict, List
import google.generativeai as genai
import random
def select_api_key(api_key: str = None) -> None:
    genai.configure(api_key=api_key or random.choice(API_KEY))
select_api_key()
print('Gemini Models: ', [m.name for m in genai.list_models()])

from utils import detect_image
from lib_chat import LRole, LType, LChat

class GeminiChat(LChat):
    NAME = 'gemini-2.0-flash-exp'

    def __init__(self, cid: int):
        super().__init__(cid)
        self.temperature: float = 1.0
        self.top_k: int = 40
        self.top_p: float = 0.95
        self.max_tokens: int = 8192

    async def get_parts(self, message: str|List[str], type: LType) -> List[object]:
        if type == LType.TEXT:
            if isinstance(message, str):
                message = [message]
            return message
        else:
            mime, data, _ = detect_image(message)
            return [{'mime_type': mime, 'data': data}]
    
    async def add_message(self, parts: List[object], role: LRole) -> None:
        match role:
            case LRole.USER:
                rolestr = 'user'
            case LRole.SYSTEM:
                rolestr = 'system'
            case LRole.MODEL:
                rolestr = 'model'
        self.history.append({'role': rolestr, 'parts': parts})
    
    async def count_tokens(self) -> int:
        #return sum(len(p) for m in self.history for p in m['parts'])
        select_api_key()
        return (await genai.GenerativeModel(self.NAME).count_tokens_async(self.history)).total_tokens

    async def cut_history(self, max_tokens: int = LChat.MAX_RCTOKENS, max_images: int = LChat.MAX_IMAGES) -> None:
        imgs, i = 0, len(self.history)
        while i > 0:
            i -= 1
            if self.history[i]['role'] == 'system':
                continue
            if isinstance(self.history[i]['parts'][0], dict):
                imgs += 1
            if imgs >= max_images:
                self.history.pop(i)
        while await self.count_tokens() > max_tokens:
            for i in range(len(self.history)):
                if self.history[i]['role'] == 'system':
                    continue
                self.history.pop(i)
                break

    async def ask(self) -> AsyncIterator[str]:
        select_api_key()
        model = genai.GenerativeModel(self.NAME,
                    system_instruction=list(map(lambda x: x['parts'][0], filter(lambda x: x['role'] == 'system', self.history))) or None)
        hist = list(filter(lambda x: x['role'] != 'system', self.history))
        response = await model.generate_content_async(
            hist,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_output_tokens=self.max_tokens),
            safety_settings=dict.fromkeys(['harassment', 'hate', 'sex', 'danger'], 'block_none'),
            stream=True
        )
        text = ''
        async for chunk in response:
            text += chunk.text
            yield text

__all__  = ['GeminiChat']
