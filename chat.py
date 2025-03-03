from dotenv import load_dotenv
load_dotenv()

import os
token = os.getenv('TGBOT_TOKEN')
assert token, 'TGBOT_TOKEN not set'

whitelist = os.getenv('TGBOT_WHITELIST')
whitelist = [w.strip() for w in whitelist.strip().split(',') if w.strip()]

keywords = os.getenv('TGBOT_KEYWORDS')
keywords = [k.strip() for k in keywords.strip().split(',') if k.strip()]

ignore_keywords = os.getenv('TGBOT_IGNORE_KEYWORDS')
ignore_keywords = [k.strip() for k in ignore_keywords.strip().split(',') if k.strip()]

import time, asyncio, functools, random, os, io, base64, json
from datetime import datetime, timedelta, timezone
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

from lib_chat import LChat, LRole, LType
from google.api_core.exceptions import InternalServerError, ResourceExhausted
from lib_gemini import GeminiChat

async def generate(message: Message, chatbot: LChat, prompt: str|List[object], role: LRole = LRole.USER) -> bool:
    if prompt:
        if isinstance(prompt, str):
            prompt = [prompt]
        await chatbot.add_message(prompt, role)
    await chatbot.cut_history()
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
    await chatbot.add_message([answer], LRole.MODEL)
    prompt, answer = repr(prompt), repr(answer)
    sp = prompt if len(prompt) < 100 else (prompt[:100] + '......')
    sa = answer if len(answer) < 100 else (answer[:100] + '......')
    ttk = len(prompt + answer)
    ctk = await chatbot.count_tokens()
    logger.info(f'Prompt cid {chatbot.cid}, t_token: {ttk}, c_tokens: {ctk}, prompt: {repr(sp)}, answer: {repr(sa)}')
    return True

chatbot_map: Dict[int, LChat] = {}
lastmsg_map: Dict[int, Message] = {}

chatmsglock_map: Dict[int, asyncio.Lock] = {}
chatmsgint_map: Dict[int, int] = {}

import atexit
@atexit.register
def cleanup():
    logger.warning('Cleanup finished')

def is_started(chat: Chat) -> bool:
    return (chat.id in chatbot_map)

def load_preset(name: str, is_group: bool = False) -> str:
    grs = 'g' if is_group else 'p'
    with open(f'presets/{grs}_{name}.txt', 'r', encoding='utf-8') as f:
        c = f.read().strip()
    return c

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    send = message.reply_text if is_group(chat) else chat.send_message
    if whitelist and chat.id not in whitelist and str(chat.id) not in whitelist:
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
            # case 'gpt' | 'openai' | 'gpt4' | 'gpt-4' | 'g4':
            #     chatcls = GPT4Chat
            # case 'gpt41' | 'gpt4(1)' | 'gpt-41' | 'g41':
            #     chatcls = GPT4Chat1
            # case 'gpt35' | 'gpt-3.5' | 'g35' | 'g3':
            #     chatcls = GPT35Chat
    except:
        pass
    chatbot_map[chat.id] = chatbot = chatcls(random.getrandbits(31), chat.id)
    if os.path.exists(f'cache/{chatbot.tid}.json'):
        os.makedirs('cache', exist_ok=True)
        with open(f'cache/{chatbot.tid}.json', 'r', encoding='utf-8') as f:
            chatbot.history = json.load(f)
    def save_cache() -> None:
        with open(f'cache/{chatbot.tid}.json', 'w', encoding='utf-8') as f:
            json.dump(chatbot.history, f, ensure_ascii=False, indent=2)
    chatbot.add_message_callback(save_cache)
    logger.info(f'/start by chatid {chat.id}, type {chatbot.NAME}, cid {chatbot.cid}; history cache {len(chatbot.history)}')
    message = await send(f'[BOT] Start {chatbot.NAME} with {len(chatbot.history)} history.')
    # if not await generate(message, chatbot, load_preset('default', is_group(chat))):
    #     del chatbot_map[chat.id]
    #     return
    if len(chatbot.history) == 0:
        await chatbot.add_message([load_preset('default', is_group(chat))], LRole.SYSTEM)
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
    message = await send(f'[BOT] Reset {chatbot.NAME} .')
    # if not await generate(message, chatbot, prompt or load_preset('default', is_group(chat)), LRole.SYSTEM if system else LRole.USER):
    #     del chatbot_map[chat.id]
    #     return
    await chatbot.add_message([prompt or load_preset('default', is_group(chat))], LRole.SYSTEM if system else LRole.USER)
    lastmsg_map[chat.id] = message

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await reset(update, context, None, True)

async def reset_full_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from datetime import date
    today = date.today()
    await reset(update, context, f'You are GPT-4, a large language model trained by OpenAI. The knowledge cutoff for this conversation is September 2021, and the current date is {today.strftime("%B %d, %Y")}', True)

async def neko_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await reset(update, context, load_preset('neko', is_group(update.effective_chat)), True)

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
    message = await send(f'[BOT] Rollbacked to {len(chatbot.history)}. Reply here.')
    lastmsg_map[chat.id] = message

async def revoke_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    if not await chatbot.revoke_message():
        await send('[BOT] Unable to revoke initial message; reset instead.')
        return
    del lastmsg_map[chat.id]
    message = await send(f'[BOT] Revoked to {len(chatbot.history)}. Reply here.')
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
    # by design, the latest message is the one to retry
    if not await chatbot.retry_message():
        await send('[BOT] Nothing to retry.')
        return
    del lastmsg_map[chat.id]
    message = await send('[BOT] Retrying...')
    await generate(message, chatbot, None)
    lastmsg_map[chat.id] = message

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
    await chatbot.add_message([system], LRole.SYSTEM)
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
    lastmsg_map[chat.id]
    chatbot_map[chat.id]
    chatmsglock_map[chat.id]
    chatmsgint_map[chat.id]
    await send('[BOT] Closed.')
    logger.info(f'/close by chatid {chat.id}, cid {chatbot.cid}')

async def handle_photos(update: Update, chatbot: LChat) -> List[dict]:
    """
    Handle all photo/sticker content by adding them as images to the chatbot context.
    If multiple photos in the message, handle each one or choose the best resolution, etc.
    Stickers are also to be treated as photos (static).
    """
    message = update.effective_message
    ret = []
    
    if message.photo:
        photo = message.photo[-1]  # Get the highest resolution photo
        file_id = photo.file_id
        file = await photo.get_file()
        buf = io.BytesIO()
        await file.download_to_memory(buf)
        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        # Add to chatbot context
        try:
            ret += await chatbot.get_parts(base64_img, LType.IMAGE)
        except:
            logger.exception('Failed to add photo to chatbot context')
        
    # Handle sticker (treat as photo)
    if message.sticker:
        # Even animated stickers are treated as photos.
        sticker = message.sticker
        file_id = sticker.file_id
        file = await sticker.get_file()
        buf = io.BytesIO()
        await file.download_to_memory(buf)
        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        try:
            ret += await chatbot.get_parts(base64_img, LType.IMAGE)
        except:
            logger.exception('Failed to add sticker to chatbot context')
    
    return ret

async def wait_for_last_message(chatid: int, interval: float, func: Callable, func1: Callable) -> None:
    await asyncio.sleep(interval)
    async with chatmsglock_map[chatid]:
        if chatmsgint_map[chatid] == 0:
            try:
                await func()
            except:
                await func1()
        else:
            await func1()

async def priv_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    prompt = update.effective_message.text or update.effective_message.caption
    send = chat.send_message
    username = update.effective_user.username or update.effective_user.full_name
    utc_now = datetime.now(timezone.utc) + timedelta(hours=8)
    time_str = utc_now.strftime('%Y-%m-%d %H:%M:%S')
    if not is_started(chat):
        await send('[BOT] Not started.')
        return
    chatbot = chatbot_map[chat.id]
    async with chatmsglock_map.setdefault(chat.id, asyncio.Lock()):
        chatmsgint_map.setdefault(chat.id, 0)
        chatmsgint_map[chat.id] += 1
    try:
        images = await handle_photos(update, chatbot)
        logger.info(f'private message by chatid {chat.id}, cid {chatbot.cid}, type {chatbot.NAME}, photos {len(images)}, prompt_len {len(prompt or [])}')
        if prompt and chat.id in lastmsg_map:
            async def _generate() -> None:
                del lastmsg_map[chat.id]
                message = await send('...')
                await generate(message, chatbot, [f"[{username}] ({time_str}): \n"] + [prompt] + images)
                lastmsg_map[chat.id] = message
            async def _append() -> None:
                await chatbot.add_message([f"[{username}] ({time_str}): \n"] + [prompt] + images, LRole.USER)
            asyncio.create_task(wait_for_last_message(chat.id, 1, _generate, _append))
        else:
            await chatbot.add_message([f"[{username}] ({time_str}): \n"] + images, LRole.USER)
    except:
        logger.exception('Failed to process private message')
    finally:
        async with chatmsglock_map[chat.id]:
            chatmsgint_map[chat.id] -= 1
    

async def group_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    prompt = update.effective_message.text or update.effective_message.caption
    #send = update.effective_message.reply_text
    send = chat.send_message
    username = update.effective_user.username or update.effective_user.full_name
    utc_now = datetime.now(timezone.utc) + timedelta(hours=8)
    time_str = utc_now.strftime('%Y-%m-%d %H:%M:%S')
    # if not update.effective_message.reply_to_message \
    #      or str(update.effective_message.reply_to_message.from_user.id) != token.split(':')[0]:
    #     return
    if not is_started(chat):
        #await send('[BOT] Not started.')
        return
    if prompt and any(k.lower() in prompt.lower() for k in ignore_keywords):
        return
    chatbot = chatbot_map[chat.id]
    async with chatmsglock_map.setdefault(chat.id, asyncio.Lock()):
        chatmsgint_map.setdefault(chat.id, 0)
        chatmsgint_map[chat.id] += 1
    try:
        images = await handle_photos(update, chatbot)
        logger.info(f'group message by chatid {chat.id}, cid {chatbot.cid}, type {chatbot.NAME}, photos {len(images)}, prompt_len {len(prompt or [])}')
        if prompt and (any(k.lower() in prompt.lower() for k in keywords) or (update.effective_message.reply_to_message and lastmsg_map[chat.id].id == update.message.reply_to_message.id)) and chat.id in lastmsg_map:
            #if lastmsg_map[chat.id].id != update.message.reply_to_message.id:
            #    return
            async def _generate() -> None:
                del lastmsg_map[chat.id]
                message = await send('...')
                await generate(message, chatbot, [f"[{username}] ({time_str}): \n"] + images + [prompt])
                lastmsg_map[chat.id] = message
            async def _append() -> None:
                await chatbot.add_message([f"[{username}] ({time_str}): \n"] + images + [prompt], LRole.USER)
            asyncio.create_task(wait_for_last_message(chat.id, 1, _generate, _append))
        else:
            await chatbot.add_message([f"[{username}] ({time_str}): \n"] + images + ([prompt] if prompt else []), LRole.USER)
    except:
        logger.exception('Failed to process group message')
    finally:
        async with chatmsglock_map[chat.id]:
            chatmsgint_map[chat.id] -= 1

if __name__ == '__main__':
    app = Application.builder().token(token)
    app = app.build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("neko", neko_command))
    app.add_handler(CommandHandler("rollback", rollback_command))
    app.add_handler(CommandHandler("revoke", revoke_command))
    app.add_handler(CommandHandler("retry", retry_command))
    app.add_handler(CommandHandler("system", system_command))
    app.add_handler(CommandHandler("reset_full", reset_full_command))
    app.add_handler(CommandHandler("close", close_command))
    app.add_handler(MessageHandler((filters.TEXT | filters.PHOTO | filters.Sticker.ALL | filters.REPLY | filters.CAPTION) & ~filters.COMMAND \
                        & filters.UpdateType.MESSAGES & filters.ChatType.PRIVATE, priv_message))
    app.add_handler(MessageHandler((filters.TEXT | filters.PHOTO | filters.Sticker.ALL | filters.REPLY | filters.CAPTION) & ~filters.COMMAND \
                        & filters.UpdateType.MESSAGE & filters.ChatType.GROUPS, group_message))
    logger.warning('TG Bot up')
    app.run_polling()
    cleanup()

