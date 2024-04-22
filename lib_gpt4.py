import json
from typing import AsyncIterator

import requests

from lib_chat import Role, Chat

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': 'application/json, text/event-stream',
    'Accept-Language': 'zh,en-US;q=0.7,en;q=0.3',
    'Content-Type': 'application/json;charset=UTF-8',
    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'TE': 'trailers',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
}

class GPT4Chat(Chat):
    NAME = 'gpt-4(caifree)'
    PROVIDER = 'https://chat.caifree.com'
    ENDPOINT = 'https://chat.caifree.com/api/openai/v1/chat/completions'
    print('GPT-4 Provider: ', PROVIDER)

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
        self.history.append({'role': rolestr, 'content': message})
    
    async def count_tokens(self) -> int:
        return sum(len(m['content']) for m in self.history)

    async def ask(self) -> AsyncIterator[str]:
        res = requests.post(self.ENDPOINT, headers=headers | {
            'Origin': self.PROVIDER,
            'Referer': self.PROVIDER + '/',
        }, json={
            'model': 'gpt-4',
            'stream': True,
            'messages': self.history,
            'temperature': self.temperature,
        }, stream=True, timeout=100)

        text = ''
        for line in res.iter_lines():
            line = line.decode('utf-8')
            if not line:
                continue
            if not line.startswith('data:'):
                raise RuntimeError(f'Unexpected response line from endpoint: {line};\n{res.text}')
            line = line[5:].strip()
            if line == '[DONE]':
                break
            try:
                line = json.loads(line)
                if len(line['choices'][0]['delta']) == 0:
                    continue
                text += line['choices'][0]['delta']['content']
                yield text
            except:
                raise RuntimeError(f'Unexpected response line from endpoint: {line};\n{res.text}')

class GPT4Chat1(GPT4Chat):
    NAME = 'gpt-4(chataionline)'
    PROVIDER = 'https://free.chataionline.top'
    ENDPOINT = 'https://free.chataionline.top/api/chat-stream'
    print('GPT-4 Provider(1): ', PROVIDER)

    async def ask(self) -> AsyncIterator[str]:
        res = requests.post(self.ENDPOINT, headers=headers | {
            'Origin': self.PROVIDER,
            'Referer': self.PROVIDER + '/',
            'path': 'v1/chat/completions'
        }, json={
            'model': 'gpt-4',
            'stream': True,
            'messages': self.history,
            'temperature': self.temperature,
        }, stream=True, timeout=100)

        if res.status_code != 200:
            raise RuntimeError(f'Unexpected response from endpoint: HTTP {res.status_code};\n{res.text}')
        
        res.encoding = 'utf-8'

        text = ''
        for line in res.iter_content(chunk_size=None, decode_unicode=True):
            if not line:
                continue
            text += line
            yield text


__all__  = ['GPT4Chat', 'GPT41Chat']
