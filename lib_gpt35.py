import json, uuid, time, random
from typing import AsyncIterator

import asyncio, logging
from threading import Thread
from curl_cffi import requests

from lib_chat import Role, Chat

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/event-stream',
    'Accept-Language': 'en-US,en;q=0.9',
    'OAI-Language': 'en-US',
    'Content-Type': 'application/json;charset=UTF-8',
    'Connection': 'keep-alive',
    'Origin': 'https://chat.openai.com',
    'Referer': 'https://chat.openai.com/',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'TE': 'trailers',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
}

def new_async_thread(wrapper, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(wrapper(*args, **kwargs))
    loop.close()

class GPT35Chat(Chat):
    NAME = 'gpt-3.5'
    PROVIDER = 'https://chat.openai.com' # Feel free to switch to any reverse proxy endpoint
    ENDPOINT = PROVIDER + '/backend-anon/conversation'
    print('GPT-3.5 Provider: ', PROVIDER)

    FTOKEN = [-1, '', '']
    LAST_ASKT = -1
    
    @staticmethod
    async def get_token() -> str:
        if GPT35Chat.FTOKEN[0] + 610 < time.time():
            await GPT35Chat.refresh()
        return GPT35Chat.FTOKEN[1:]
    
    @staticmethod
    async def keep_refresh() -> None:
        while True:
            await GPT35Chat.refresh()
            while GPT35Chat.LAST_ASKT + 3600 < time.time():
                await asyncio.sleep(random.randint(300, 600))
    
    Thread(target=new_async_thread, args=(keep_refresh,), daemon=True).start()

    def __init__(self, cid: int):
        super().__init__(cid)
        self.temperature: float = 1.0

    async def add_message(self, message: str, role: Role) -> None:
        match role:
            case Role.USER:
                rolestr = 'user'
            case Role.SYSTEM:
                rolestr = 'system'
            case Role.MODEL:
                rolestr = 'assistant'
        self.history.append({'author': {'role': rolestr},
            'content': {'content_type': 'text', 'parts': [message]},
            'id': str(uuid.uuid4()),
            'metadata': {},
        })
    
    async def count_tokens(self) -> int:
        return sum(len(m['content']['parts'][0]) for m in self.history)

    @staticmethod
    async def refresh() -> None:
        try:
            asess = requests.AsyncSession(impersonate="chrome120")
            res = await asess.post(GPT35Chat.PROVIDER + '/backend-anon/sentinel/chat-requirements', headers=headers | {
                'OAI-Device-Id': (did := str(uuid.uuid4())),
            }, json={}, timeout=10)
            res.raise_for_status()
            GPT35Chat.FTOKEN = [time.time(), did, res.json()['token']]
        except Exception as e:
            logging.exception('Error getting OAID for ChatGPT-3.5')

    async def ask(self) -> AsyncIterator[str]:
        GPT35Chat.LAST_ASKT = time.time()
        token = await self.get_token()
        asess = requests.AsyncSession(impersonate="chrome120")
        res = await asess.post(self.ENDPOINT, headers=headers | {
            'OAI-Device-Id': token[0],
            'Openai-Sentinel-Chat-Requirements-Token': token[1],
        }, json={
            'action': 'next',
            'messages': self.history,
            'parent_message_id': str(uuid.uuid4()),
            'model': 'text-davinci-002-render-sha',
            'timezone_offset_min': -180,
            'suggestions': [],
            'history_and_training_disabled': True,
            'conversation_mode': {'kind': 'primary_assistant'},
            'force_nulligen': False,
            'force_paragen': False,
            'force_paragen_model_slug': '',
            'force_rate_limit': False,
            'websocket_request_id': str(uuid.uuid4()),
        }, stream=True, timeout=100, impersonate="chrome120")

        async for line in res.aiter_lines():
            line = line.decode('utf-8')
            if not line:
                continue
            if not line.startswith('data:'):
                #raise RuntimeError(f'Unexpected response line from endpoint: {line};\n{res.text}')
                print(line)
                continue
            line = line[5:].strip()
            if line == '[DONE]':
                break
            try:
                line = json.loads(line)
                if 'type' in line and line['type'] == 'moderation':
                    continue
                if len(line['message']['content']['parts']) == 0:
                    continue
                yield line['message']['content']['parts'][0]
            except:
                raise RuntimeError(f'Unexpected response line from endpoint: {line};\n{res.text}')


__all__  = ['GPT35Chat']
