from dotenv import load_dotenv
load_dotenv()

import os
API_KEY = os.getenv('DEEPSEEK_API_KEY')
assert API_KEY, 'DEEPSEEK_API_KEY not set'
API_KEY = [k.strip() for k in API_KEY.split() if k.strip()]

import openai
import random
from typing import AsyncIterator,override
def getclient() -> openai.AsyncOpenAI:
    client = openai.AsyncOpenAI(api_key=random.choice(API_KEY), base_url='https://api.deepseek.com')
    return client

import tiktoken
from lib_chat import LChat

class DeepseekChat(LChat):
    NAME = 'deepseek-reasoner'
    MAX_RCTOKENS = 60000
    MAX_IMAGES = 0

    def __init__(self, cid: int, tid: int):
        super().__init__(cid, tid)
    
    async def convert_hist(self, history) -> list[object]:
        hist = []
        prev_role = None
        for c in history:
            role = c['role']
            if role == 'system':
                role = 'user'
            if role == 'model':
                role = 'assistant'
            if prev_role == role:
                hist[-1]['parts'].extend(c['parts'])
            else:
                hist.append({'role': role, 'parts': c['parts']})
                prev_role = role
        for c in hist:
            c['parts'] = ['\n ---------- \n'.join(filter(lambda p: isinstance(p, str), c['parts']))]
        return hist

    async def count_tokens(self) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return sum(len(encoding.encode(p))+1 if isinstance(p, str) else 0 for m in await self.convert_hist(self.history) for p in m['parts'])

    async def ask(self) -> AsyncIterator[str]:
        client = getclient()
        addp = [{'role': 'user', 'parts': ['[Strictly follow system prompt and reply]:\n']}] # optional
        hist = []
        for c in await self.convert_hist(self.history + addp):
            role = c['role']
            for p in c['parts']:
                if isinstance(p, str):
                    hist.append({'role': role, 'content': p})
        response = await client.chat.completions.create(
            model=self.NAME,
            messages=hist,
            stream=True
        )
        is_reasoning = False
        async for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                if not is_reasoning:
                   is_reasoning = True
                   yield (False, '[reasoning] ', )
                yield (False, chunk.choices[0].delta.reasoning_content, )
            elif chunk.choices[0].delta.content:
                if is_reasoning:
                    is_reasoning = False
                    yield (False, ' [/reasoning]\n\n', )
                yield chunk.choices[0].delta.content

__all__ = ['DeepseekChat']
