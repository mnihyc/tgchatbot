API_KEY = 'YOUR_GEMINI_API_KEY_HERE'

from typing import AsyncIterator
import google.generativeai as genai
genai.configure(api_key=API_KEY)
print('Gemini Models: ', [m.name for m in genai.list_models()])

from lib_chat import Role, Chat

class GeminiChat(Chat):
    NAME = 'gemini-1.5-pro'

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
                rolestr = 'model'
        self.history.append({'role': rolestr, 'parts': [message]})
    
    async def count_tokens(self) -> int:
        return sum(len(p) for m in self.history for p in m['parts'])

    async def ask(self) -> AsyncIterator[str]:
        model = genai.GenerativeModel('gemini-1.5-pro-latest',
                    system_instruction=list(map(lambda x: x['parts'][0], filter(lambda x: x['role'] == 'system', self.history))) or None)
        # compability issues; select last line from system prompt as first user input
        hist = list(filter(lambda x: x['role'] != 'system', self.history))
        if self.history[0]['role'] == 'system':
            hist.insert(0, {'role': 'user', 'parts': [self.history[0]['parts'][0].split('\n')[-1]]})
        response = await model.generate_content_async(
            hist,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=8192,
                temperature=self.temperature),
            safety_settings=dict.fromkeys(['harassment', 'hate', 'sex', 'danger'], 'block_none'),
            stream=True
        )
        text = ''
        async for chunk in response:
            text += chunk.text
            yield text

__all__  = ['GeminiChat']
