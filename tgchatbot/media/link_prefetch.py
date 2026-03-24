from __future__ import annotations

"""Lightweight URL prefetch used at ingest time.

Telegram does not provide webpage content to bots as part of normal message updates.
When enabled, this module fetches a small amount of page metadata/content so the
chatbot can react to shared links without spending a full tool round on web search.
"""

from dataclasses import dataclass
import asyncio
from html import unescape
from html.parser import HTMLParser
import ipaddress
import re
import socket
from urllib.parse import urlparse

import httpx

from tgchatbot.config import TelegramConfig
from tgchatbot.domain.models import MessagePart, PartKind

_URL_RE = re.compile(r'https?://[^\s<>()\[\]{}"\']+')
_WS_RE = re.compile(r'\s+')


class _PreviewHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.title_parts: list[str] = []
        self.description: str | None = None
        self.body_text: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        attrs = {k.lower(): (v or '') for k, v in attrs}
        if tag.lower() == 'title':
            self.in_title = True
        elif tag.lower() == 'meta':
            name = attrs.get('name', '').lower()
            prop = attrs.get('property', '').lower()
            if name == 'description' or prop == 'og:description':
                content = attrs.get('content', '').strip()
                if content and not self.description:
                    self.description = content

    def handle_endtag(self, tag: str):
        if tag.lower() == 'title':
            self.in_title = False

    def handle_data(self, data: str):
        text = _normalize(data)
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
        elif len(self.body_text) < 60:
            self.body_text.append(text)


@dataclass(slots=True)
class LinkPreviewData:
    url: str
    title: str | None = None
    description: str | None = None
    snippet: str | None = None
    content_type: str | None = None


def _normalize(text: str) -> str:
    return _WS_RE.sub(' ', unescape(text or '')).strip()


def _host_is_blocked(host: str) -> bool:
    lowered = (host or '').strip().lower().strip('.')
    if not lowered:
        return True
    if lowered in {'localhost'} or lowered.endswith('.localhost') or lowered.endswith('.local'):
        return True
    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified)


async def _url_is_safe(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {'http', 'https'}:
        return False
    host = parsed.hostname or ''
    if _host_is_blocked(host):
        return False
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, parsed.port or (443 if parsed.scheme == 'https' else 80), type=socket.SOCK_STREAM)
    except Exception:
        return False
    for family, _socktype, _proto, _canonname, sockaddr in infos:
        ip_text = sockaddr[0] if isinstance(sockaddr, tuple) and sockaddr else None
        if not ip_text:
            continue
        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return False
    return True


def extract_urls(text: str, *, max_urls: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in _URL_RE.findall(text or ''):
        url = raw.rstrip('.,!?;:')
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= max_urls:
            break
    return out


async def fetch_link_previews(text: str, *, mode: str, telegram: TelegramConfig) -> list[LinkPreviewData]:
    urls = extract_urls(text, max_urls=max(0, telegram.link_prefetch_max_urls))
    if not urls or mode == 'off':
        return []
    timeout = httpx.Timeout(telegram.link_prefetch_timeout_s, connect=min(telegram.link_prefetch_timeout_s, 3.0))
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, headers={'User-Agent': 'tgchatbot-link-prefetch/1.0'}) as client:
        results: list[LinkPreviewData] = []
        for url in urls:
            try:
                if not await _url_is_safe(url):
                    continue
                preview = await _fetch_one(client, url, mode=mode, max_chars=telegram.link_prefetch_max_chars)
            except Exception:
                continue
            if preview:
                results.append(preview)
        return results


async def _fetch_one(client: httpx.AsyncClient, url: str, *, mode: str, max_chars: int) -> LinkPreviewData | None:
    current_url = url
    for _ in range(5):
        if not await _url_is_safe(current_url):
            return None
        response = await client.get(current_url)
        if response.is_redirect:
            location = response.headers.get('location', '').strip()
            if not location:
                return None
            current_url = str(response.url.join(location))
            continue
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        parsed = urlparse(str(response.url))
        host = parsed.netloc or parsed.path
        if 'text/html' in content_type:
            body = response.text[: max(max_chars * 2, 4000)]
            parser = _PreviewHTMLParser()
            parser.feed(body)
            title = _normalize(' '.join(parser.title_parts)) or None
            description = _normalize(parser.description or '') or None
            snippet = None
            if mode == 'snippet':
                body_text = _normalize(' '.join(parser.body_text))
                if body_text:
                    snippet = body_text[:max_chars]
            return LinkPreviewData(url=str(response.url), title=title or host, description=description, snippet=snippet, content_type=content_type)
        if content_type.startswith('text/'):
            text = _normalize(response.text)
            if not text:
                return None
            return LinkPreviewData(url=str(response.url), title=host, description=text[:max_chars], snippet=(text[:max_chars] if mode == 'snippet' else None), content_type=content_type)
        return LinkPreviewData(url=str(response.url), title=host, description=f'Linked resource ({content_type or "unknown content"})', content_type=content_type)
    return None


def previews_to_parts(previews: list[LinkPreviewData], *, mode: str) -> list[MessagePart]:
    parts: list[MessagePart] = []
    for preview in previews:
        lines = ['[Link prefetched, content:]', f'URL: {preview.url}']
        if preview.title:
            lines.append(f'Title: {preview.title}')
        if preview.description:
            lines.append(f'Description: {preview.description}')
        if mode == 'snippet' and preview.snippet:
            lines.append(f'Snippet: {preview.snippet}')
        parts.append(MessagePart(kind=PartKind.TEXT, text='\n'.join(lines), remote_sync=False, origin='auto_note'))
    return parts
