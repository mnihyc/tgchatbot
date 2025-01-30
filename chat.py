import os
token = os.getenv('TGBOT_TOKEN')
assert(token, 'TGBOT_TOKEN not set')

whitelist = os.getenv('TGBOT_WHITELIST')

import time, asyncio, functools, random, os
from typing import AsyncIterator, Callable, Dict, List
from annotated_types import T
def awaitify(sync_func):
    @functools.wraps(sync_func)
    async def async_func(*args, **kwargs):
        return sync_func(*args, **kwargs)
    return async_func

import logging
logging.basicConfig(level = logging.WARNING, format = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from telegram import Bot, Chat, Message, Update
from telegram.constants import ChatType, ParseMode, ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

def is_group(chat: Chat) -> bool:
    return (str(chat.type) in (str(ChatType.GROUP), str(ChatType.SUPERGROUP), ))

def is_private(chat: Chat) -> bool:
    return (str(chat.type) == str(ChatType.PRIVATE))

async def keep_typing_while(message: Message, func: Callable[[None], T]) -> T:
    event = asyncio.Event()

    async def keep_typing() -> None:
        while True:
            await message.get_bot().send_chat_action(message.chat.id, ChatAction.TYPING)
            try:
                await asyncio.wait_for(event.wait(), timeout=4)
            except asyncio.TimeoutError:
                continue
            break

    async def executor() -> T:
        try:
            ret = await func()
        except Exception as e:
            event.set() # prevent keep_typing from hanging
            raise       # safe to re-raise now
        event.set()
        return ret

    ret = await asyncio.gather(
        keep_typing(),
        executor(),
    )
    return ret[1]

from lib_chat import Chat, Role
from google.api_core.exceptions import InternalServerError, ResourceExhausted
from lib_gemini import GeminiChat
from lib_gpt4 import GPT4Chat, GPT4Chat1
from lib_gpt35 import GPT35Chat

async def generate(message: Message, chatbot: Chat, prompt: str, role: Role = Role.USER) -> bool:
    await chatbot.add_message(prompt, role)
    async def _ask() -> str:
        prev = sent = ''
        last = time.perf_counter()
        async for data in chatbot.ask():
            if data.strip() == prev.strip():
                continue
            prev = data
            now = time.perf_counter()
            if now-last>5.0 or (now-last>2.5 and len(data)-len(sent)>25):
                last = now
                await message.edit_text(sent := data)
        if prev != sent:
            await message.edit_text(prev)
        return prev
    for retries in range(3):
        try:
            try:
                answer = await keep_typing_while(message, _ask)
            except (RuntimeError, InternalServerError, ResourceExhausted) as e:  # StopAsyncIteration unable to catch
                if retries == 2: raise  # maximum retries
                logger.warning(f'Retrying query for {type(e)}, cid {chatbot.cid}\nREASON: ' + repr(e)[:min(100,len(repr(e)))])
                await message.edit_text(f'[BOT] Retrying for {type(e)} ... ({retries})\nREASON: ' + repr(e)[:min(100,len(repr(e)))])
                await asyncio.sleep(0.1 + retries * 2)
                continue
        except Exception as e:
            logger.exception('ERROR RUNNING QUERY.')
            await message.edit_text('[BOT] EXCEPTION: ' + repr(e)[:min(100,len(repr(e)))])
            await chatbot.pop_message()
            return False
        break
    if len(answer) == 0:
        logger.warning('Got empty response. Possibly moderation.')
        await message.edit_text('[BOT] EMPTY RESPONSE. SKIP.')
        return False
    await chatbot.add_message(answer, Role.MODEL)
    sp = prompt if len(prompt) < 100 else (prompt[:100] + '......')
    sa = answer if len(answer) < 100 else (answer[:100] + '......')
    ttk = len(prompt + answer)
    ctk = await chatbot.count_tokens()
    logger.info(f'Prompt cid {chatbot.cid}, t_token: {ttk}, c_tokens: {ctk}, prompt: {repr(sp)}, answer: {repr(sa)}')
    return True

chatbot_map: Dict[int, Chat] = {}
lastmsg_map: Dict[int, Message] = {}

import atexit
@atexit.register
def cleanup():
    logger.warning('Cleanup finished')

def is_started(chat: Chat) -> bool:
    return (chat.id in chatbot_map)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    send = message.reply_text if is_group(chat) else chat.send_message
    if chat.id not in whitelist:
        logger.info(f'restricted /start by chatid {chat.id}')
        await send('[BOT] Whitelist restricted.')
        return
    if is_started(chat):
        await send('[BOT] Already started.')
        return
    chatcls = GeminiChat
    try:
        match context.args[0].strip():
            case 'google' | 'gemini' | 'gemini-pro' | 'go' | 'ge':
                chatcls = GeminiChat
            case 'gpt' | 'openai' | 'gpt4' | 'gpt-4' | 'g4':
                chatcls = GPT4Chat
            case 'gpt41' | 'gpt4(1)' | 'gpt-41' | 'g41':
                chatcls = GPT4Chat1
            case 'gpt35' | 'gpt-3.5' | 'g35' | 'g3':
                chatcls = GPT35Chat
    except:
        pass
    chatbot_map[chat.id] = chatbot = chatcls(random.getrandbits(31))
    logger.info(f'/start by chatid {chat.id}, type {chatbot.NAME}, cid {chatbot.cid}')
    message = await send(f'[BOT] Starting {chatbot.NAME} ...')
    if not await generate(message, chatbot, 'Hello'):
        del chatbot_map[chat.id]
        return
    lastmsg_map[chat.id] = message

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str, system: bool = False) -> None:
    chat = update.effective_chat
    message = update.effective_message
    send = message.reply_text if is_group(chat) else chat.send_message
    if not is_started(chat):
        await send('[BOT] Not started.')
        return
    chatbot = chatbot_map[chat.id]
    if chat.id not in lastmsg_map:
        await send('[BOT] Still updating.')
        return
    del lastmsg_map[chat.id]
    logger.info(f'characteristic /reset by chatid {chat.id}, cid {chatbot.cid}, type {chatbot.NAME}')
    await chatbot.clear_message()
    message = await send(f'[BOT] Resetting {chatbot.NAME} ...')
    if not await generate(message, chatbot, prompt, Role.SYSTEM if system else Role.USER):
        del chatbot_map[chat.id]
        return
    lastmsg_map[chat.id] = message

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await reset(update, context, 'Hello')

async def reset_full_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from datetime import date
    today = date.today()
    await reset(update, context, f'You are GPT-4, a large language model trained by OpenAI. The knowledge cutoff for this conversation is September 2021, and the current date is {today.strftime("%B %d, %Y")}', True)

async def rollback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    send = message.reply_text if is_group(chat) else chat.send_message
    if not is_started(chat):
        await send('[BOT] Not started.')
        return
    chatbot = chatbot_map[chat.id]
    if chat.id not in lastmsg_map:
        await send('[BOT] Still updating.')
        return
    if not await chatbot.rollback_message():
        await send('[BOT] Unable to revoke initial message; reset instead.')
        return
    del lastmsg_map[chat.id]
    message = await send('[BOT] Rollbacked last conversation. Reply here.')
    lastmsg_map[chat.id] = message

async def retry_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    send = message.reply_text if is_group(chat) else chat.send_message
    if not is_started(chat):
        await send('[BOT] Not started.')
        return
    chatbot = chatbot_map[chat.id]
    if chat.id not in lastmsg_map:
        await send('[BOT] Still updating.')
        return
    # by design, there will be at least 2 messages in gvcs
    chatbot.pop_message()
    del lastmsg_map[chat.id]
    message = await send('[BOT] Retrying...')
    await generate(message, chatbot, None)
    lastmsg_map[chat.id] = message

async def neko_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with open('presets/neko.txt', 'r', encoding='utf-8') as f:
        c = f.read().strip()
    await reset(update, context, c, True)

async def nekoss_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    with open('presets/nekoss.txt', 'r', encoding='utf-8') as f:
        c = f.read().strip()
    await reset(update, context, c, True)

async def system_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    send = message.reply_text if is_group(chat) else chat.send_message
    if not is_started(chat):
        await send('[BOT] Not started.')
        return
    chatbot = chatbot_map[chat.id]
    system = message.text.partition('/system')[2].strip()
    if len(system) == 0:
        await send('[BOT] Empty system message.')
        return
    await chatbot.add_message(system, Role.SYSTEM)
    message = await send('[BOT] System message appended.')
    lastmsg_map[chat.id] = message

async def close_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    send = message.reply_text if is_group(chat) else chat.send_message
    if not is_started(chat):
        await send('[BOT] Not started.')
        return
    chatbot = chatbot_map[chat.id]
    if chat.id not in lastmsg_map:
        await send('[BOT] Still updating.')
        #return
    del lastmsg_map[chat.id]
    del chatbot_map[chat.id]
    await send('[BOT] Closed.')
    logger.info(f'/close by chatid {chat.id}, cid {chatbot.cid}')

async def set_temperature_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    send = message.reply_text if is_group(chat) else chat.send_message
    if not is_started(chat):
        await send('[BOT] Not started.')
        return
    chatbot = chatbot_map[chat.id]
    try:
        temperature = float(context.args[0])
    except:
        await send('[BOT] Parameter: 0.0 <= temperature <= 1.0')
        return
    chatbot.temperature = temperature
    message = await send('[BOT] Temperature set.')
    lastmsg_map[chat.id] = message

async def priv_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    prompt = update.effective_message.text
    send = chat.send_message
    if not is_started(chat):
        await send('[BOT] Not started.')
        return
    chatbot = chatbot_map[chat.id]
    if chat.id not in lastmsg_map:
        await send('[BOT] Still updating.')
        return
    del lastmsg_map[chat.id]
    message = await send('[BOT] Generating...')
    await generate(message, chatbot, prompt)
    lastmsg_map[chat.id] = message

async def group_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    prompt = update.effective_message.text
    send = update.effective_message.reply_text
    if not update.effective_message.reply_to_message \
         or str(update.effective_message.reply_to_message.from_user.id) != token.split(':')[0]:
        return
    if not is_started(chat):
        await send('[BOT] Not started.')
        return
    chatbot = chatbot_map[chat.id]
    if chat.id not in lastmsg_map:
        await send('[BOT] Still updating.')
        return
    #if lastmsg_map[chat.id].id != update.message.reply_to_message.id:
    #    return
    del lastmsg_map[chat.id]
    message = await send('[BOT] Generating...')
    await generate(message, chatbot, prompt)
    lastmsg_map[chat.id] = message

if __name__ == '__main__':
    app = Application.builder().token(token)
    app = app.build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("rollback", rollback_command))
    app.add_handler(CommandHandler("retry", retry_command))
    app.add_handler(CommandHandler("neko", neko_command))
    app.add_handler(CommandHandler("nekoss", nekoss_command))
    app.add_handler(CommandHandler("system", system_command))
    app.add_handler(CommandHandler("reset_full", reset_full_command))
    app.add_handler(CommandHandler("close", close_command))
    app.add_handler(CommandHandler("set_temperature", set_temperature_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND \
                        & filters.UpdateType.MESSAGE & filters.ChatType.PRIVATE \
                        & ~filters.REPLY & ~filters.FORWARDED, priv_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND \
                        & filters.UpdateType.MESSAGE & filters.ChatType.GROUPS \
                        & filters.REPLY, group_message))
    logger.warning('TG Bot up')
    app.run_polling()
    cleanup()

