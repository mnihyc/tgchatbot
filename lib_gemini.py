from dotenv import load_dotenv
load_dotenv()

import os
API_KEY = os.getenv('GEMINI_API_KEY')
assert API_KEY, 'GEMINI_API_KEY not set'
API_KEY = [k.strip() for k in API_KEY.split() if k.strip()]

from typing import AsyncIterator, Dict, List, override
from google import genai
from google.genai import types
import random
def getclient() -> genai.Client:
    client = genai.Client(api_key=random.choice(API_KEY))
    return client
print('Initialized Gemini Models: ', [m.name for m in getclient().models.list()])

import tiktoken, base64
from lib_chat import LChat

class GeminiChat(LChat):
    NAME = 'gemini-2.0-flash'
    MAX_RCTOKENS = 1000000
    MAX_IMAGES = 3000

    def __init__(self, cid: int, tid: int):
        super().__init__(cid, tid)
        self.temperature: float = 1.0
        self.top_k: int = 40
        self.top_p: float = 0.95
        self.max_tokens: int = 8192
    
    async def count_tokens(self) -> int:
        #return sum(len(p) for m in self.history for p in m['parts'])
        #return (await getclient().aio.models.count_tokens(model=self.NAME, contents=self.history)).total_tokens
        encoding = tiktoken.get_encoding("cl100k_base")
        return sum(len(encoding.encode(p))+1 if isinstance(p, str) else 258*4 for m in self.history for p in m['parts'])

    async def ask(self) -> AsyncIterator[str]:
        client = getclient()
        hist = []
        for c in list(filter(lambda x: x['role'] != 'system', self.history)):
            role = c['role']
            if role == 'system':
                continue
            parts = []
            for p in c['parts']:
                if isinstance(p, str):
                    parts.append(types.Part.from_text(text=p))
                elif isinstance(p, dict):
                    parts.append(types.Part.from_bytes(mime_type=p['mime_type'], data=base64.b64decode(p['data'])))
            hist.append(types.Content(role=role, parts=parts))
        hist.append(types.Content(role='user', parts=[types.Part.from_text(text='[Strictly follow system prompt and reply]:\n')])) # optional
        async for chunk in await client.aio.models.generate_content_stream(
            model = self.NAME,
            contents = hist,
            config = genai.types.GenerateContentConfig(
                system_instruction='\n'.join(list(map(lambda x: x['parts'][0], filter(lambda x: x['role'] == 'system', self.history)))) or None,
                candidate_count=1,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_output_tokens=self.max_tokens,
                safety_settings=[
                    types.SafetySetting(
                        category='HARM_CATEGORY_HATE_SPEECH',
                        threshold='BLOCK_NONE',
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_DANGEROUS_CONTENT',
                        threshold='BLOCK_NONE',
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_HARASSMENT',
                        threshold='BLOCK_NONE',
                    ),
                    types.SafetySetting(
                        category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                        threshold='BLOCK_NONE',
                    ),
                ],
            ),
        ):
            if chunk.text:
                yield chunk.text

__all__  = ['GeminiChat']
