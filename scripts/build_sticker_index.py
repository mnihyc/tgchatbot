#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import base64
import gc
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from paddleocr import PaddleOCR
import av

SUPPORTED_EXTS = {'.webp', '.webm', '.gif', '.png', '.jpg', '.jpeg', '.mp4'}
ANIMATED_EXTS = {'webm', 'gif', 'mp4'}

from tgchatbot.stickers.schema import (
    PRIMARY_ANALYSIS_PROMPT as PROMPT,
    REVIEW_PROMPT,
    STICKER_SCHEMA_VERSION,
    compose_caption_semantic_text,
    compose_sticker_semantic_text,
    compose_style_text,
    semantic_signature,
    strict_response_schema,
    strict_review_schema,
)

SCHEMA = strict_response_schema()
REVIEW_SCHEMA = strict_review_schema()


@dataclass(slots=True)
class SourceSticker:
    path: Path
    relative_path: str
    source_pack_id: str | None
    source_format: str
    source_size_bytes: int
    source_mtime_ns: int
    sha1: str


@dataclass(slots=True)
class BuiltStickerRow:
    row: dict[str, Any]
    doc: dict[str, Any]
    caption_embed_text: str
    sticker_embed_text: str
    style_feature_tokens: list[str]
    visual_features: np.ndarray


@dataclass(slots=True)
class BuildWorkerConfig:
    max_frames: int
    ocr_lang: str
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    embedding_model: str
    embedding_dimensions: int
    openai_read_timeout: float
    openai_connect_timeout: float
    openai_retries: int


_WORKER_CONFIG: BuildWorkerConfig | None = None
_WORKER_OCR: 'PaddleOCRExtractor | None' = None
_WORKER_LLM: 'OpenAITextClient | None' = None


def _close_worker_clients() -> None:
    global _WORKER_LLM
    if _WORKER_LLM is not None:
        try:
            _WORKER_LLM.close()
        except Exception:
            pass
        _WORKER_LLM = None


def _init_worker(config: BuildWorkerConfig) -> None:
    global _WORKER_CONFIG, _WORKER_OCR, _WORKER_LLM
    _WORKER_CONFIG = config
    _WORKER_OCR = PaddleOCRExtractor(lang=config.ocr_lang)
    if not _WORKER_OCR.available:
        raise RuntimeError('PaddleOCR is required for build_sticker_index.py. Install paddlepaddle and paddleocr first.')
    _WORKER_LLM = OpenAITextClient(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        model=config.openai_model,
        embedding_model=config.embedding_model,
        embedding_dimensions=config.embedding_dimensions,
        read_timeout=config.openai_read_timeout,
        connect_timeout=config.openai_connect_timeout,
        retries=config.openai_retries,
    )
    atexit.register(_close_worker_clients)


def _ensure_worker_state() -> tuple['PaddleOCRExtractor', 'OpenAITextClient', BuildWorkerConfig]:
    if _WORKER_CONFIG is None or _WORKER_OCR is None or _WORKER_LLM is None:
        raise RuntimeError('Sticker build worker was used before initialization.')
    return _WORKER_OCR, _WORKER_LLM, _WORKER_CONFIG


class PaddleOCRExtractor:
    def __init__(self, lang: str = "ch") -> None:
        self.lang = lang
        self._ocr = None

    @property
    def available(self) -> bool:
        return PaddleOCR is not None

    def _ensure(self) -> None:
        if self._ocr is not None:
            return
        if PaddleOCR is None:
            raise RuntimeError(
                "PaddleOCR is not installed. Install paddlepaddle>=3 and paddleocr first."
            )
        self._ocr = PaddleOCR(
            lang=self.lang,
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def extract(self, image: Image.Image) -> dict[str, Any]:
        self._ensure()
        rgb = image.convert("RGB")

        parsed = {"lines": [], "joined_text": "", "confidence": 0.0, "coverage_ratio": 0.0}

        # Prefer file-path predict() for PaddleOCR 3.x stability
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            rgb.save(tmp_path)

            # Try predict() first
            try:
                result = self._ocr.predict(tmp_path)
                parsed = _parse_paddle_predict_result(result)
            except Exception:
                parsed = {"lines": [], "joined_text": "", "confidence": 0.0, "coverage_ratio": 0.0}

            # Empty parse is not success; retry legacy .ocr()
            if not parsed.get("lines"):
                try:
                    arr = np.array(rgb)
                    result = self._ocr.ocr(arr, cls=True)
                    parsed = _parse_paddle_ocr_result(result)
                except Exception:
                    parsed = {"lines": [], "joined_text": "", "confidence": 0.0, "coverage_ratio": 0.0}

            # Tiny text retry: upscale 2x then 3x
            if not parsed.get("lines"):
                for factor in (2, 3):
                    up = rgb.resize(
                        (rgb.width * factor, rgb.height * factor),
                        Image.Resampling.LANCZOS,
                    )
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
                        tmp2_path = tmp2.name
                    try:
                        up.save(tmp2_path)
                        result = self._ocr.predict(tmp2_path)
                        parsed = _parse_paddle_predict_result(result)
                        if parsed.get("lines"):
                            break
                    except Exception:
                        pass
                    finally:
                        try:
                            os.remove(tmp2_path)
                        except OSError:
                            pass
                    if parsed.get("lines"):
                        break

            parsed["image_size"] = {"width": rgb.width, "height": rgb.height}
            parsed["coverage_ratio"] = _coverage_ratio(
                parsed.get("lines", []),
                rgb.width,
                rgb.height,
            )
            return parsed
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            try:
                rgb.close()
            except Exception:
                pass


class OpenAITextClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        embedding_model: str,
        embedding_dimensions: int,
        read_timeout: float,
        connect_timeout: float,
        retries: int,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.retries = max(0, int(retries))
        self.client = httpx.Client(
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        )

    def close(self) -> None:
        self.client.close()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.post(f'{self.base_url}{path}', json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                last_exc = exc
                if status not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt >= self.retries:
                    raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.TransportError) as exc:
                last_exc = exc
                if attempt >= self.retries:
                    raise
            time.sleep(min(8.0, 1.5 * (2 ** attempt)))
        assert last_exc is not None
        raise last_exc

    def _responses_json(self, *, prompt: str, schema_name: str, schema: dict[str, Any], frame_payloads: list[dict[str, str]]) -> dict[str, Any]:
        parts: list[dict[str, Any]] = [{'type': 'input_text', 'text': prompt}]
        for frame in frame_payloads:
            parts.append({'type': 'input_image', 'image_url': f"data:{frame['mime']};base64,{frame['data']}", 'detail': 'auto'})
        payload = {
            'model': self.model,
            'store': False,
            'input': [{'role': 'user', 'content': parts}],
            'text': {'format': {'type': 'json_schema', 'name': schema_name, 'schema': schema, 'strict': True}},
        }
        body = self._post_json('/responses', payload)
        text = body.get('output_text', '')
        if not text:
            pieces: list[str] = []
            for item in body.get('output', []):
                if item.get('type') == 'message':
                    for content in item.get('content', []):
                        if content.get('type') == 'output_text':
                            pieces.append(content.get('text', ''))
            text = ''.join(pieces)
        return json.loads(text)

    def analyze_primary(self, *, relative_path: str, source_format_name: str, ocr_summary: dict[str, Any], frame_payloads: list[dict[str, str]]) -> dict[str, Any]:
        prompt = (
            PROMPT
            + f"\nSticker path: {relative_path}"
            + f"\nSticker file format: {source_format_name}"
            + f"\nOCR overlay text: {ocr_summary.get('joined_text', '')}"
            + f"\nOCR lines JSON: {json.dumps(ocr_summary.get('lines', []), ensure_ascii=False)}"
            + f"\nOCR confidence: {ocr_summary.get('confidence', 0.0)}"
            + f"\nOCR coverage ratio: {ocr_summary.get('coverage_ratio', 0.0)}"
        )
        return self._responses_json(prompt=prompt, schema_name='sticker_semantics_primary', schema=SCHEMA, frame_payloads=frame_payloads)

    def validate_semantics(self, *, relative_path: str, source_format_name: str, ocr_summary: dict[str, Any], candidate: dict[str, Any], frame_payloads: list[dict[str, str]]) -> dict[str, Any]:
        prompt = (
            REVIEW_PROMPT
            + f"\nSticker path: {relative_path}"
            + f"\nSticker file format: {source_format_name}"
            + f"\nOCR overlay text: {ocr_summary.get('joined_text', '')}"
            + f"\nOCR lines JSON: {json.dumps(ocr_summary.get('lines', []), ensure_ascii=False)}"
            + f"\nOCR confidence: {ocr_summary.get('confidence', 0.0)}"
            + f"\nOCR coverage ratio: {ocr_summary.get('coverage_ratio', 0.0)}"
            + f"\nCandidate analysis JSON: {json.dumps(candidate, ensure_ascii=False)}"
        )
        review = self._responses_json(prompt=prompt, schema_name='sticker_semantics_review', schema=REVIEW_SCHEMA, frame_payloads=frame_payloads)
        result = review.get('result')
        if not isinstance(result, dict):
            raise RuntimeError('Semantic review did not return a valid result object.')
        result.setdefault('_semantic_review', {
            'alignment_score': float(review.get('alignment_score', 0.0) or 0.0),
            'caption_authority': float(review.get('caption_authority', 0.0) or 0.0),
            'notes': str(review.get('notes', '') or ''),
        })
        return result

    def analyze(self, *, relative_path: str, source_format_name: str, ocr_summary: dict[str, Any], frame_payloads: list[dict[str, str]]) -> dict[str, Any]:
        primary = self.analyze_primary(relative_path=relative_path, source_format_name=source_format_name, ocr_summary=ocr_summary, frame_payloads=frame_payloads)
        revised = self.validate_semantics(relative_path=relative_path, source_format_name=source_format_name, ocr_summary=ocr_summary, candidate=primary, frame_payloads=frame_payloads)
        return revised

    def embed_many(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dimensions), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), max(1, batch_size)):
            payload = {
                'model': self.embedding_model,
                'input': texts[start:start + max(1, batch_size)],
                'dimensions': self.embedding_dimensions,
            }
            body = self._post_json('/embeddings', payload)
            rows = body['data']
            batch = np.asarray([row['embedding'] for row in rows], dtype=np.float32)
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            batches.append(batch / norms)
        return np.vstack(batches)

def _parse_paddle_predict_result(result: Any) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []

    if result is None:
        return {"lines": [], "joined_text": "", "confidence": 0.0, "coverage_ratio": 0.0}

    for item in result:
        # PaddleOCR 3.x commonly stores actual OCR payload in `.res`
        if hasattr(item, "res"):
            obj = item.res
        elif isinstance(item, dict) and "res" in item:
            obj = item["res"]
        else:
            obj = item

        if not isinstance(obj, dict):
            continue

        rec_texts = list(obj.get("rec_texts", []) or [])
        rec_scores = list(obj.get("rec_scores", []) or [])
        rec_polys = list(obj.get("rec_polys", []) or [])

        for idx, text in enumerate(rec_texts):
            content = str(text).strip()
            if not content:
                continue
            score = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
            box = rec_polys[idx] if idx < len(rec_polys) else None
            if hasattr(box, "tolist"):
                box = box.tolist()
            lines.append(
                {
                    "text": content,
                    "score": score,
                    "box": box,
                }
            )

    return _ocr_summary(lines)


def _parse_paddle_ocr_result(result: Any) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []
    if not result:
        return {'lines': [], 'joined_text': '', 'confidence': 0.0, 'coverage_ratio': 0.0}
    outer = result[0] if isinstance(result, list) and result else result
    for item in outer or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text = str(item[1][0]).strip()
        score = float(item[1][1]) if len(item[1]) > 1 else 0.0
        if text:
            lines.append({'text': text, 'score': score, 'box': item[0]})
    return _ocr_summary(lines)


def _ocr_summary(lines: list[dict[str, Any]]) -> dict[str, Any]:
    if not lines:
        return {'lines': [], 'joined_text': '', 'confidence': 0.0, 'coverage_ratio': 0.0}
    joined_text = ' | '.join(line['text'] for line in lines)
    confidence = sum(float(line['score']) for line in lines) / max(1, len(lines))
    return {'lines': lines, 'joined_text': joined_text, 'confidence': confidence, 'coverage_ratio': 0.0}


def _coverage_ratio(lines: list[dict[str, Any]], width: int, height: int) -> float:
    if not lines or width <= 0 or height <= 0:
        return 0.0
    total = 0.0
    for line in lines:
        total += _box_area(line.get('box'))
    return max(0.0, min(1.0, total / float(width * height)))


def _box_area(box: Any) -> float:
    if not box:
        return 0.0
    try:
        pts = np.asarray(box, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)
    except Exception:
        return 0.0


def _iter_stickers(root: Path) -> list[SourceSticker]:
    rows: list[SourceSticker] = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        relative = path.relative_to(root).as_posix()
        parts = Path(relative).parts
        source_pack_id = parts[0] if len(parts) > 1 else None
        stat = path.stat()
        rows.append(SourceSticker(
            path=path,
            relative_path=relative,
            source_pack_id=source_pack_id,
            source_format=path.suffix.lower().lstrip('.'),
            source_size_bytes=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns,
            sha1=hashlib.sha1(path.read_bytes()).hexdigest(),
        ))
    return rows


def _extract_frames(path: Path, max_frames: int) -> list[Image.Image]:
    data = path.read_bytes()
    try:
        img = Image.open(io.BytesIO(data))
        total_frames = getattr(img, 'n_frames', 1)
        if total_frames <= 1:
            return [img.copy()]
        targets = [int(i * (total_frames - 1) / max(1, max_frames - 1)) for i in range(min(max_frames, total_frames))]
        frames: list[Image.Image] = []
        for idx in sorted(set(targets)):
            img.seek(idx)
            frames.append(img.copy())
        return frames
    except Exception:
        if av is None:
            raise
        container = av.open(io.BytesIO(data))
        stream = container.streams.video[0]
        frames = [frame.to_image() for frame in container.decode(stream)][: max_frames * 4]
        container.close()
        if len(frames) <= max_frames:
            return frames
        return [frames[int(i * (len(frames) - 1) / max(1, max_frames - 1))] for i in range(max_frames)]


def _encode_frame(frame: Image.Image, max_size: int = 768) -> dict[str, str]:
    working = frame.copy()
    working.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    if working.mode in ('RGBA', 'LA') or ('transparency' in working.info):
        working.convert('RGBA').save(buf, format='PNG', optimize=True)
        mime = 'image/png'
    else:
        working.convert('RGB').save(buf, format='JPEG', quality=88, optimize=True)
        mime = 'image/jpeg'
    return {'mime': mime, 'data': base64.b64encode(buf.getvalue()).decode('utf-8')}


def _normalize_text(value: str) -> str:
    return ' '.join(str(value or '').replace('\n', ' ').replace('\t', ' ').split()).strip()


def _script_langs(text: str) -> list[str]:
    langs: list[str] = []
    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
        langs.append('chinese')
    if any('\u3040' <= ch <= '\u30ff' for ch in text):
        langs.append('japanese')
    if any('\uac00' <= ch <= '\ud7af' for ch in text):
        langs.append('korean')
    if any('A' <= ch <= 'Z' or 'a' <= ch <= 'z' for ch in text):
        langs.append('latin')
    return langs


def _visual_features(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert('RGB').resize((128, 128)), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    grad_x = np.diff(gray, axis=1, append=gray[:, -1:])
    grad_y = np.diff(gray, axis=0, append=gray[-1:, :])
    edge_density = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = float(np.mean(maxc - minc))
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return np.asarray([brightness, contrast, edge_density, saturation, colorfulness], dtype=np.float32)


def _scale_label(value: str) -> float:
    mapping = {
        'none': 0.0, 'very_low': 0.5, 'low': 1.0, 'mild': 1.0, 'medium': 2.0, 'moderate': 2.0,
        'high': 3.0, 'strong': 3.0, 'very_high': 4.0, 'extreme': 4.0,
        'light': 1.0, 'heavy': 3.0, 'minimal': 0.5, 'prominent': 3.0,
    }
    return mapping.get(_normalize_text(value).lower(), 0.0)


def _semantic_force_score(semantics: dict[str, Any]) -> float:
    caption_card = semantics.get('caption_card', {}) or {}
    subtle_cue_card = semantics.get('subtle_cue_card', {}) or {}
    sticker_card = semantics.get('sticker_card', {}) or {}
    style_card = semantics.get('style_card', {}) or {}
    review = semantics.get('_semantic_review', {}) or {}
    score = 0.0
    score += 0.45 * float(caption_card.get('harshness_level', 0) or 0)
    score += 0.30 * float(caption_card.get('meme_dependence_level', 0) or 0)
    score += 0.22 * float(caption_card.get('intimacy_level', 0) or 0)
    score += 0.18 * _scale_label(caption_card.get('caption_reply_force', ''))
    score += 0.12 * _scale_label(caption_card.get('caption_irony_strength', ''))
    score += 0.22 * _scale_label(subtle_cue_card.get('visual_reply_force', ''))
    score += 0.14 * _scale_label(subtle_cue_card.get('visual_irony_strength', ''))
    score += 0.22 * _scale_label(sticker_card.get('sticker_reply_force', ''))
    score += 0.14 * _scale_label(sticker_card.get('sticker_irony_strength', ''))
    score += 0.20 * _scale_label(style_card.get('style_text_prominence', ''))
    score += 0.26 * float(review.get('caption_authority', 0.0) or 0.0)
    for key in (
        'caption_literal_meaning', 'caption_pragmatic_meaning', 'caption_discourse_role', 'caption_social_stance',
        'caption_reply_force', 'caption_emotional_valence', 'caption_irony_strength',
    ):
        value = _normalize_text(caption_card.get(key, ''))
        if len(value) >= 6:
            score += 0.10
    for key in (
        'dominant_signal', 'micro_expression', 'eye_signal', 'pose_signal', 'visual_social_stance',
        'visual_reply_force', 'visual_emotional_valence', 'visual_irony_strength',
    ):
        value = _normalize_text(subtle_cue_card.get(key, ''))
        if len(value) >= 6:
            score += 0.08
    for key in (
        'sticker_reaction_type', 'fused_pragmatic_meaning', 'sticker_visual_emotion', 'sticker_reply_force',
        'sticker_emotional_valence', 'sticker_irony_strength',
    ):
        value = _normalize_text(sticker_card.get(key, ''))
        if len(value) >= 6:
            score += 0.08
    return score


def _estimate_caption_dominance_score(*, overlay_text: str, confidence: float, coverage_ratio: float, line_count: int, semantics: dict[str, Any]) -> int:
    overlay_text = _normalize_text(overlay_text)
    if not overlay_text:
        return 0
    score = 0.0
    text_len = len(overlay_text.replace(' ', '').replace('|', ''))
    score += min(1.4, math.log2(max(2, text_len))) * 0.55
    score += min(1.2, 0.4 * max(1, line_count))
    score += max(0.0, min(1.4, confidence * 1.5))
    score += max(0.0, min(1.6, coverage_ratio * 9.0))
    score += min(1.8, _semantic_force_score(semantics) * 0.35)
    rounded = int(round(score))
    return max(1, min(4, rounded))


def _card_texts(semantics: dict[str, Any]) -> tuple[str, str, str]:
    caption_text = compose_caption_semantic_text(semantics)
    sticker_text = compose_sticker_semantic_text(semantics)
    style_text = compose_style_text(semantics)
    return caption_text, sticker_text, style_text


def _style_feature_tokens(*, source_pack_id: str | None, style_text: str, style_card: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for value in [style_text, style_card.get('style_rendering_type', ''), style_card.get('line_weight', ''), style_card.get('style_palette_family', ''), style_card.get('meme_intensity', ''), style_card.get('style_text_prominence', ''), style_card.get('style_character_family', '')]:
        for token in _normalize_text(str(value)).lower().replace(';', ' ').replace(',', ' ').split():
            if token:
                tokens.append(token)
    if source_pack_id:
        tokens.append(f'packhint:{source_pack_id.lower()}')
    return tokens


def _style_vectors(rows: list[BuiltStickerRow]) -> tuple[np.ndarray, dict[str, int]]:
    vocab: dict[str, int] = {}
    for row in rows:
        for token in row.style_feature_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    dim = len(vocab) + 5
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    for row_idx, built in enumerate(rows):
        counts: dict[int, float] = {}
        for token in built.style_feature_tokens:
            idx = vocab[token]
            counts[idx] = counts.get(idx, 0.0) + 1.0
        for idx, value in counts.items():
            matrix[row_idx, idx] = value
        matrix[row_idx, len(vocab):] = built.visual_features
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, vocab


def _kmeans_cluster(matrix: np.ndarray, desired_clusters: int) -> np.ndarray:
    n = matrix.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    k = max(1, min(desired_clusters, n))
    rng = np.random.default_rng(42)
    if n <= k:
        return np.arange(n, dtype=np.int32)
    centroids = matrix[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(18):
        sims = matrix @ centroids.T
        new_labels = np.argmax(sims, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            members = matrix[labels == idx]
            if len(members) == 0:
                centroids[idx] = matrix[rng.integers(0, n)]
                continue
            centroid = members.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[idx] = centroid / norm if norm > 0 else centroid
    return labels


def _cluster_target(count: int) -> int:
    if count <= 16:
        return max(1, count)
    return max(8, min(64, int(round(math.sqrt(count * 1.6)))))


def _schema_sql() -> str:
    return '''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS stickers (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            sticker_id TEXT NOT NULL UNIQUE,
            relative_path TEXT NOT NULL UNIQUE,
            source_format TEXT NOT NULL,
            source_pack_id TEXT,
            source_pack_hash INTEGER NOT NULL,
            summary TEXT NOT NULL,
            preview_text TEXT NOT NULL,
            emoji TEXT,
            caption_mode TEXT NOT NULL,
            source_overlay_text TEXT NOT NULL,
            source_overlay_text_normalized TEXT NOT NULL,
            source_overlay_languages TEXT NOT NULL,
            source_ocr_confidence REAL NOT NULL,
            source_ocr_confidence_bucket INTEGER NOT NULL,
            caption_dominance_score INTEGER NOT NULL,
            caption_meaning_en TEXT NOT NULL,
            caption_meaning_zh TEXT NOT NULL,
            caption_semantic_text TEXT NOT NULL,
            sticker_semantic_text TEXT NOT NULL,
            style_text TEXT NOT NULL,
            caption_card_json TEXT NOT NULL,
            subtle_cue_card_json TEXT NOT NULL,
            sticker_card_json TEXT NOT NULL,
            style_card_json TEXT NOT NULL,
            semantic_signature TEXT NOT NULL,
            style_cluster TEXT NOT NULL,
            style_cluster_id INTEGER NOT NULL,
            harshness_level INTEGER NOT NULL,
            intimacy_level INTEGER NOT NULL,
            meme_dependence_level INTEGER NOT NULL,
            nsfw_stub_flag INTEGER NOT NULL,
            animated INTEGER NOT NULL,
            source_size_bytes INTEGER NOT NULL,
            source_mtime_ns INTEGER NOT NULL,
            sha1 TEXT NOT NULL,
            selection_notes TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
    '''

def _drop_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        '''
        DROP TABLE IF EXISTS stickers;
        DROP TABLE IF EXISTS meta;
        '''
    )
    con.commit()

def _table_columns(con: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        rows = con.execute(f'PRAGMA table_info({table_name})').fetchall()
    except sqlite3.DatabaseError:
        return set()
    return {str(row[1]) for row in rows}

def _ensure_schema(con: sqlite3.Connection, *, rebuild: bool) -> None:
    if rebuild:
        _drop_schema(con)
    con.executescript(_schema_sql())
    con.commit()
    required = {
        'sticker_id', 'relative_path', 'style_cluster', 'style_cluster_id',
        'caption_card_json', 'subtle_cue_card_json', 'sticker_card_json', 'style_card_json',
        'metadata_json', 'selection_notes', 'sha1',
    }
    existing = _table_columns(con, 'stickers')
    missing = required - existing
    if missing:
        raise RuntimeError(f'Index database schema is missing columns: {sorted(missing)}. Re-run with --rebuild.')

def _existing_sticker_state(con: sqlite3.Connection) -> dict[str, str]:
    if not _table_columns(con, 'stickers'):
        return {}
    return {
        str(row['relative_path']): str(row['sha1'])
        for row in con.execute('SELECT relative_path, sha1 FROM stickers')
    }

def _upsert(con: sqlite3.Connection, row: dict[str, Any]) -> None:
    con.execute(
        '''
        INSERT OR REPLACE INTO stickers (
            sticker_id, relative_path, source_format, source_pack_id, source_pack_hash, summary, preview_text, emoji, caption_mode,
            source_overlay_text, source_overlay_text_normalized, source_overlay_languages, source_ocr_confidence, source_ocr_confidence_bucket, caption_dominance_score,
            caption_meaning_en, caption_meaning_zh, caption_semantic_text, sticker_semantic_text, style_text,
            caption_card_json, subtle_cue_card_json, sticker_card_json, style_card_json, semantic_signature, style_cluster, style_cluster_id,
            harshness_level, intimacy_level, meme_dependence_level, nsfw_stub_flag,
            animated, source_size_bytes, source_mtime_ns, sha1, selection_notes, metadata_json
        ) VALUES (
            :sticker_id, :relative_path, :source_format, :source_pack_id, :source_pack_hash, :summary, :preview_text, :emoji, :caption_mode,
            :source_overlay_text, :source_overlay_text_normalized, :source_overlay_languages, :source_ocr_confidence, :source_ocr_confidence_bucket, :caption_dominance_score,
            :caption_meaning_en, :caption_meaning_zh, :caption_semantic_text, :sticker_semantic_text, :style_text,
            :caption_card_json, :subtle_cue_card_json, :sticker_card_json, :style_card_json, :semantic_signature, :style_cluster, :style_cluster_id,
            :harshness_level, :intimacy_level, :meme_dependence_level, :nsfw_stub_flag,
            :animated, :source_size_bytes, :source_mtime_ns, :sha1, :selection_notes, :metadata_json
        )
        ''',
        row,
    )


def _build_row(source: SourceSticker, semantics: dict[str, Any], ocr_summary: dict[str, Any], visual_features: np.ndarray) -> BuiltStickerRow:
    caption_card = semantics['caption_card']
    subtle_cue_card = semantics['subtle_cue_card']
    sticker_card = semantics['sticker_card']
    style_card = semantics['style_card']
    overlay_raw = _normalize_text(ocr_summary.get('joined_text', ''))
    source_overlay_languages = _script_langs(overlay_raw)
    confidence = float(ocr_summary.get('confidence', 0.0) or 0.0)
    coverage = float(ocr_summary.get('coverage_ratio', 0.0) or 0.0)
    caption_dominance_score = _estimate_caption_dominance_score(
        overlay_text=overlay_raw,
        confidence=confidence,
        coverage_ratio=coverage,
        line_count=len(ocr_summary.get('lines', [])),
        semantics=semantics,
    )
    caption_mode = caption_card.get('caption_mode', 'mixed')
    caption_authority = float((semantics.get('_semantic_review', {}) or {}).get('caption_authority', 0.0) or 0.0)
    if overlay_raw and (caption_dominance_score >= 3 or caption_authority >= 0.7):
        caption_mode = 'caption_dominant'
    elif caption_dominance_score == 0 and caption_authority < 0.35:
        caption_mode = 'visual_dominant'
    elif caption_mode not in {'caption_dominant', 'mixed', 'visual_dominant'}:
        caption_mode = 'mixed'
    caption_card['caption_mode'] = caption_mode
    caption_meaning_en = _normalize_text(caption_card.get('caption_meaning_en', ''))
    caption_meaning_zh = _normalize_text(caption_card.get('caption_meaning_zh', ''))
    caption_text, sticker_text, style_text = _card_texts(semantics)
    sticker_id = hashlib.sha1(source.relative_path.encode('utf-8')).hexdigest()[:16]
    source_pack_hash = int(hashlib.sha1((source.source_pack_id or '').encode('utf-8')).hexdigest()[:12], 16)
    sem_sig = semantic_signature(semantics)
    metadata = {
        'schema_version': STICKER_SCHEMA_VERSION,
        'ocr_lines': ocr_summary.get('lines', []),
        'ocr_coverage_ratio': coverage,
        'source_relative_path': source.relative_path,
        'caption_mode': caption_mode,
        'visual_features': [float(x) for x in visual_features],
        'semantic_review': semantics.get('_semantic_review', {}),
        'semantic_signature': sem_sig,
    }
    row = {
        'sticker_id': sticker_id,
        'relative_path': source.relative_path,
        'source_format': source.source_format,
        'source_pack_id': source.source_pack_id,
        'source_pack_hash': source_pack_hash,
        'summary': _normalize_text(semantics['summary']),
        'preview_text': _normalize_text(semantics['preview_text']),
        'emoji': semantics['emoji'],
        'caption_mode': caption_mode,
        'source_overlay_text': overlay_raw,
        'source_overlay_text_normalized': overlay_raw,
        'source_overlay_languages': json.dumps(source_overlay_languages, ensure_ascii=False),
        'source_ocr_confidence': confidence,
        'source_ocr_confidence_bucket': max(0, min(4, int(round(confidence * 4)))),
        'caption_dominance_score': caption_dominance_score,
        'caption_meaning_en': caption_meaning_en,
        'caption_meaning_zh': caption_meaning_zh,
        'caption_semantic_text': caption_text,
        'sticker_semantic_text': sticker_text,
        'style_text': style_text,
        'caption_card_json': json.dumps(caption_card, ensure_ascii=False),
        'subtle_cue_card_json': json.dumps(subtle_cue_card, ensure_ascii=False),
        'sticker_card_json': json.dumps(sticker_card, ensure_ascii=False),
        'style_card_json': json.dumps(style_card, ensure_ascii=False),
        'semantic_signature': sem_sig,
        'style_cluster': '',
        'style_cluster_id': -1,
        'harshness_level': int(caption_card['harshness_level']),
        'intimacy_level': int(caption_card['intimacy_level']),
        'meme_dependence_level': int(caption_card['meme_dependence_level']),
        'nsfw_stub_flag': 0,
        'animated': 1 if source.source_format in ANIMATED_EXTS else 0,
        'source_size_bytes': source.source_size_bytes,
        'source_mtime_ns': source.source_mtime_ns,
        'sha1': source.sha1,
        'selection_notes': _normalize_text(semantics['selection_notes']),
        'metadata_json': json.dumps(metadata, ensure_ascii=False),
    }
    doc = {
        'sticker_id': sticker_id,
        'relative_path': source.relative_path,
        'source_pack_id': source.source_pack_id or '',
        'source_overlay_text_normalized': row['source_overlay_text_normalized'],
        'caption_meaning_en': row['caption_meaning_en'],
        'caption_meaning_zh': row['caption_meaning_zh'],
        'caption_semantic_text': row['caption_semantic_text'],
        'sticker_semantic_text': row['sticker_semantic_text'],
        'style_text': row['style_text'],
        'caption_dominance_score': row['caption_dominance_score'],
        'source_ocr_confidence_bucket': row['source_ocr_confidence_bucket'],
        'harshness_level': row['harshness_level'],
        'intimacy_level': row['intimacy_level'],
        'meme_dependence_level': row['meme_dependence_level'],
        'semantic_signature': row['semantic_signature'],
        'style_cluster': '',
        'style_cluster_id': -1,
        'source_pack_hash': row['source_pack_hash'],
        'animated': row['animated'],
        'nsfw_stub_flag': 0,
        'preview_text': row['preview_text'],
        'selection_notes': row['selection_notes'],
    }
    caption_embed_text = ' ; '.join(filter(None, [row['caption_meaning_en'], row['caption_meaning_zh'], row['caption_semantic_text'], row['selection_notes']]))
    sticker_embed_text = ' ; '.join(filter(None, [row['summary'], row['sticker_semantic_text'], row['style_text'], row['selection_notes']]))
    return BuiltStickerRow(
        row=row,
        doc=doc,
        caption_embed_text=caption_embed_text,
        sticker_embed_text=sticker_embed_text,
        style_feature_tokens=_style_feature_tokens(source_pack_id=source.source_pack_id, style_text=row['style_text'], style_card=style_card),
        visual_features=visual_features,
    )



def _process_source_sticker(source: SourceSticker) -> BuiltStickerRow:
    ocr, llm, worker_config = _ensure_worker_state()
    frames: list[Image.Image] = []
    frame_payloads: list[dict[str, str]] = []
    try:
        frames = _extract_frames(source.path, max_frames=max(1, worker_config.max_frames))
        frame_payloads = [_encode_frame(frame) for frame in frames[: max(1, worker_config.max_frames)]]
        primary_frame = frames[0]
        visual_features = _visual_features(primary_frame)
        try:
            ocr_summary = ocr.extract(primary_frame)
        except Exception as exc:
            raise RuntimeError(f'PaddleOCR failed for {source.relative_path}: {exc}') from exc
        try:
            semantics = llm.analyze(
                relative_path=source.relative_path,
                source_format_name=source.source_format,
                ocr_summary=ocr_summary,
                frame_payloads=frame_payloads,
            )
        except Exception as exc:
            raise RuntimeError(f'LLM semantic analysis failed for {source.relative_path}: {exc}') from exc
        return _build_row(source, semantics, ocr_summary, visual_features)
    finally:
        for frame in frames:
            try:
                frame.close()
            except Exception:
                pass
        frame_payloads.clear()
        gc.collect()


def _default_worker_count() -> int:
    cpu_total = os.cpu_count() or 1
    return max(1, min(3, cpu_total))


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def _print_progress(*, completed: int, total: int, ok: int, skipped: int, failed: int, started_at: float, last_path: str = '') -> None:
    elapsed = max(0.001, time.time() - started_at)
    rate = completed / elapsed if completed else 0.0
    remaining = max(0, total - completed)
    eta = remaining / rate if rate > 0 else 0.0
    suffix = f' | {last_path}' if last_path else ''
    print(
        f'[{completed}/{total}] ok={ok} skipped={skipped} failed={failed} rate={rate:.2f}/s eta={_format_duration(eta)}{suffix}',
        flush=True,
    )


def _append_failure(failures_path: Path, *, source: SourceSticker, exc: Exception) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'relative_path': source.relative_path,
        'source_format': source.source_format,
        'sha1': source.sha1,
        'error': str(exc),
    }
    with failures_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


def _load_all_built_rows(con: sqlite3.Connection) -> list[BuiltStickerRow]:
    rows: list[BuiltStickerRow] = []
    cursor = con.execute('SELECT * FROM stickers ORDER BY relative_path')
    for db_row in cursor:
        row = dict(db_row)
        metadata = json.loads(row['metadata_json']) if row.get('metadata_json') else {}
        style_card = json.loads(row['style_card_json']) if row.get('style_card_json') else {}
        visual_raw = metadata.get('visual_features', [0.0] * 5)
        visual_list = list(visual_raw)[:5]
        visual_list.extend([0.0] * max(0, 5 - len(visual_list)))
        visual_features = np.asarray(visual_list, dtype=np.float32)
        doc = {
            'sticker_id': row['sticker_id'],
            'relative_path': row['relative_path'],
            'source_pack_id': row['source_pack_id'] or '',
            'source_overlay_text_normalized': row['source_overlay_text_normalized'],
            'caption_meaning_en': row['caption_meaning_en'],
            'caption_meaning_zh': row['caption_meaning_zh'],
            'caption_semantic_text': row['caption_semantic_text'],
            'sticker_semantic_text': row['sticker_semantic_text'],
            'style_text': row['style_text'],
            'caption_dominance_score': row['caption_dominance_score'],
            'source_ocr_confidence_bucket': row['source_ocr_confidence_bucket'],
            'harshness_level': row['harshness_level'],
            'intimacy_level': row['intimacy_level'],
            'meme_dependence_level': row['meme_dependence_level'],
            'semantic_signature': row['semantic_signature'],
            'style_cluster': row['style_cluster'],
            'style_cluster_id': row['style_cluster_id'],
            'source_pack_hash': row['source_pack_hash'],
            'animated': row['animated'],
            'nsfw_stub_flag': row['nsfw_stub_flag'],
            'preview_text': row['preview_text'],
            'selection_notes': row['selection_notes'],
        }
        caption_embed_text = ' ; '.join(filter(None, [row['caption_meaning_en'], row['caption_meaning_zh'], row['caption_semantic_text'], row['selection_notes']]))
        sticker_embed_text = ' ; '.join(filter(None, [row['summary'], row['sticker_semantic_text'], row['style_text'], row['selection_notes']]))
        rows.append(
            BuiltStickerRow(
                row=row,
                doc=doc,
                caption_embed_text=caption_embed_text,
                sticker_embed_text=sticker_embed_text,
                style_feature_tokens=_style_feature_tokens(source_pack_id=row['source_pack_id'], style_text=row['style_text'], style_card=style_card),
                visual_features=visual_features,
            )
        )
    return rows


def _build_rows_parallel(
    *,
    sources: list[SourceSticker],
    worker_config: BuildWorkerConfig,
    workers: int,
    con: sqlite3.Connection,
    append_mode: bool,
    failures_path: Path,
    progress_every: int,
    max_in_flight: int,
    worker_max_tasks: int,
) -> dict[str, int]:
    total = len(sources)
    if total == 0:
        return {'total': 0, 'ok': 0, 'skipped': 0, 'failed': 0}
    existing = _existing_sticker_state(con) if append_mode else {}
    pending: list[SourceSticker] = []
    skipped = 0
    for source in sources:
        if existing.get(source.relative_path) == source.sha1:
            skipped += 1
            continue
        pending.append(source)
    ok = 0
    failed = 0
    completed = skipped
    started_at = time.time()
    if skipped:
        _print_progress(completed=completed, total=total, ok=ok, skipped=skipped, failed=failed, started_at=started_at, last_path='resume-skip')
    if not pending:
        return {'total': total, 'ok': 0, 'skipped': skipped, 'failed': 0}

    normalized_workers = max(1, workers)
    progress_every = max(1, progress_every)
    commit_every = max(1, min(25, progress_every))

    def handle_success(built: BuiltStickerRow) -> None:
        nonlocal ok
        _upsert(con, built.row)
        ok += 1
        if ok % commit_every == 0:
            con.commit()

    def report(last_path: str, *, force: bool = False) -> None:
        if force or completed == total or completed <= 10 or completed % progress_every == 0:
            _print_progress(completed=completed, total=total, ok=ok, skipped=skipped, failed=failed, started_at=started_at, last_path=last_path)

    if normalized_workers == 1:
        _init_worker(worker_config)
        try:
            for source in pending:
                try:
                    built = _process_source_sticker(source)
                    handle_success(built)
                except Exception as exc:
                    failed += 1
                    _append_failure(failures_path, source=source, exc=exc)
                    completed += 1
                    report(f'FAILED {source.relative_path}', force=True)
                    continue
                completed += 1
                report(source.relative_path)
        finally:
            _close_worker_clients()
            con.commit()
        return {'total': total, 'ok': ok, 'skipped': skipped, 'failed': failed}

    ctx = mp.get_context('spawn')
    max_pending = max(normalized_workers, max_in_flight)
    source_iter = iter(pending)
    with ProcessPoolExecutor(
        max_workers=normalized_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(worker_config,),
        max_tasks_per_child=max(1, worker_max_tasks),
    ) as executor:
        futures: dict[Any, SourceSticker] = {}

        def submit_next() -> bool:
            try:
                source = next(source_iter)
            except StopIteration:
                return False
            futures[executor.submit(_process_source_sticker, source)] = source
            return True

        for _ in range(min(max_pending, len(pending))):
            if not submit_next():
                break

        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                source = futures.pop(future)
                try:
                    built = future.result()
                    handle_success(built)
                except Exception as exc:
                    failed += 1
                    _append_failure(failures_path, source=source, exc=exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
                    completed += 1
                    report(f'FAILED {source.relative_path}', force=True)
                    submit_next()
                    continue
                completed += 1
                report(source.relative_path)
                submit_next()

    con.commit()
    return {'total': total, 'ok': ok, 'skipped': skipped, 'failed': failed}


def _write_validation_report(*, built_rows: list[BuiltStickerRow], output_path: Path, embedding_dimensions: int, embedding_model: str) -> None:
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'schema_version': STICKER_SCHEMA_VERSION,
        'count': len(built_rows),
        'embedding_model': embedding_model,
        'embedding_dimensions': embedding_dimensions,
        'caption_modes': {},
        'caption_dominance_score_histogram': {},
        'overlay_present': 0,
        'source_ocr_confidence': {'mean': 0.0, 'min': 0.0, 'max': 0.0},
        'style_clusters_expected': _cluster_target(len(built_rows)),
        'semantic_review': {'mean_alignment': 0.0, 'mean_caption_authority': 0.0},
        'missing': {
            'caption_meaning_en': 0,
            'caption_meaning_zh': 0,
            'caption_semantic_text': 0,
            'sticker_semantic_text': 0,
            'style_text': 0,
            'semantic_signature': 0,
        },
        'samples': [],
    }
    if not built_rows:
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return
    confidences: list[float] = []
    alignments: list[float] = []
    authorities: list[float] = []
    for built in built_rows:
        row = built.row
        report['caption_modes'][row['caption_mode']] = report['caption_modes'].get(row['caption_mode'], 0) + 1
        report['caption_dominance_score_histogram'][str(row['caption_dominance_score'])] = report['caption_dominance_score_histogram'].get(str(row['caption_dominance_score']), 0) + 1
        if row['source_overlay_text']:
            report['overlay_present'] += 1
        confidences.append(float(row['source_ocr_confidence']))
        for key in ('caption_meaning_en', 'caption_meaning_zh', 'caption_semantic_text', 'sticker_semantic_text', 'style_text', 'semantic_signature'):
            if not row[key]:
                report['missing'][key] += 1
        review = row['metadata_json'] and json.loads(row['metadata_json']).get('semantic_review', {}) or {}
        if review:
            alignments.append(float(review.get('alignment_score', 0.0) or 0.0))
            authorities.append(float(review.get('caption_authority', 0.0) or 0.0))
        if len(report['samples']) < 12:
            report['samples'].append({
                'relative_path': row['relative_path'],
                'caption_mode': row['caption_mode'],
                'caption_dominance_score': row['caption_dominance_score'],
                'source_overlay_text': row['source_overlay_text'],
                'preview_text': row['preview_text'],
                'semantic_signature': row['semantic_signature'],
                'style_cluster': row['style_cluster'],
            })
    report['source_ocr_confidence'] = {'mean': float(np.mean(confidences)), 'min': float(np.min(confidences)), 'max': float(np.max(confidences))}
    if alignments:
        report['semantic_review']['mean_alignment'] = float(np.mean(alignments))
    if authorities:
        report['semantic_review']['mean_caption_authority'] = float(np.mean(authorities))
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build sticker metadata, OCR output, semantic cards, Tantivy docs, and local embeddings.')
    parser.add_argument('--stickers-dir', default='./data/stickers')
    parser.add_argument('--index-db', default='./data/sticker_index.sqlite3')
    parser.add_argument('--max-frames', type=int, default=4)
    parser.add_argument('--ocr-lang', default='ch')
    parser.add_argument('--workers', type=int, default=int(os.getenv('STICKER_BUILD_WORKERS', str(_default_worker_count()))))
    parser.add_argument('--rebuild', action='store_true', help='Drop the current index and rebuild from scratch instead of resuming/skipping completed stickers.')
    parser.add_argument('--progress-every', type=int, default=int(os.getenv('STICKER_BUILD_PROGRESS_EVERY', '10')))
    parser.add_argument('--max-in-flight', type=int, default=int(os.getenv('STICKER_BUILD_MAX_IN_FLIGHT', '0')))
    parser.add_argument('--worker-max-tasks', type=int, default=int(os.getenv('STICKER_BUILD_WORKER_MAX_TASKS', '100')))
    parser.add_argument('--openai-base-url', default=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    parser.add_argument('--openai-api-key', default=os.getenv('OPENAI_API_KEY', ''))
    parser.add_argument('--openai-model', default=os.getenv('STICKER_TAGGING_MODEL', 'gpt-5'))
    parser.add_argument('--embedding-model', default=os.getenv('STICKER_EMBEDDING_MODEL', 'text-embedding-3-large'))
    parser.add_argument('--embedding-dimensions', type=int, default=int(os.getenv('STICKER_EMBEDDING_DIMENSIONS', '1024')))
    parser.add_argument('--openai-read-timeout', type=float, default=float(os.getenv('OPENAI_READ_TIMEOUT', '300')))
    parser.add_argument('--openai-connect-timeout', type=float, default=float(os.getenv('OPENAI_CONNECT_TIMEOUT', '30')))
    parser.add_argument('--openai-retries', type=int, default=int(os.getenv('OPENAI_REQUEST_RETRIES', '2')))
    args = parser.parse_args()

    sticker_root = Path(args.stickers_dir).expanduser().resolve()
    index_db = Path(args.index_db).expanduser().resolve()
    index_db.parent.mkdir(parents=True, exist_ok=True)
    docs_jsonl = index_db.parent / 'tantivy_docs.jsonl'
    manifest_path = index_db.parent / 'embeddings_manifest.json'
    caption_npy = index_db.parent / 'caption_embeddings.npy'
    sticker_npy = index_db.parent / 'sticker_embeddings.npy'
    style_clusters_path = index_db.parent / 'style_clusters.json'
    validation_report_path = index_db.parent / 'build_validation_report.json'
    failures_log_path = index_db.parent / 'build_failures.jsonl'

    if PaddleOCR is None:
        raise RuntimeError('PaddleOCR is required for build_sticker_index.py. Install paddlepaddle and paddleocr first.')
    if not args.openai_api_key:
        raise RuntimeError('OPENAI_API_KEY / --openai-api-key is required. LLM semantic analysis and embeddings are mandatory.')

    sources = _iter_stickers(sticker_root)
    worker_config = BuildWorkerConfig(
        max_frames=max(1, args.max_frames),
        ocr_lang=args.ocr_lang,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        openai_read_timeout=args.openai_read_timeout,
        openai_connect_timeout=args.openai_connect_timeout,
        openai_retries=args.openai_retries,
    )

    con = sqlite3.connect(index_db, timeout=30.0)
    con.row_factory = sqlite3.Row
    _ensure_schema(con, rebuild=args.rebuild)
    con.execute('INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', ('schema_version', STICKER_SCHEMA_VERSION))
    con.commit()

    max_in_flight = args.max_in_flight if args.max_in_flight > 0 else max(1, args.workers * 2)
    build_stats = _build_rows_parallel(
        sources=sources,
        worker_config=worker_config,
        workers=args.workers,
        con=con,
        append_mode=not args.rebuild,
        failures_path=failures_log_path,
        progress_every=args.progress_every,
        max_in_flight=max_in_flight,
        worker_max_tasks=args.worker_max_tasks,
    )

    built_rows = _load_all_built_rows(con)
    style_matrix, vocab = _style_vectors(built_rows)
    labels = _kmeans_cluster(style_matrix, _cluster_target(len(built_rows)))
    cluster_summary: dict[str, dict[str, Any]] = {}
    for built, label in zip(built_rows, labels, strict=True):
        cluster_name = f'style_cluster_{int(label):03d}'
        built.row['style_cluster'] = cluster_name
        built.row['style_cluster_id'] = int(label)
        built.doc['style_cluster'] = cluster_name
        built.doc['style_cluster_id'] = int(label)
        cluster = cluster_summary.setdefault(cluster_name, {'count': 0, 'examples': []})
        cluster['count'] += 1
        if len(cluster['examples']) < 8:
            cluster['examples'].append(built.row['relative_path'])
        _upsert(con, built.row)

    docs = [built.doc for built in built_rows]
    caption_embed_inputs = [built.caption_embed_text for built in built_rows]
    sticker_embed_inputs = [built.sticker_embed_text for built in built_rows]
    sticker_ids = [built.row['sticker_id'] for built in built_rows]

    build_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    con.execute('INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', ('schema_version', STICKER_SCHEMA_VERSION))
    con.execute('INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', ('needs_rebuild', '0'))
    con.execute('INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', ('build_stack', 'paddleocr+openai+tantivy+embeddings'))
    con.execute('INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', ('build_id', build_id))
    con.execute('INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', ('embedding_dimensions', str(args.embedding_dimensions)))
    con.execute('INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', ('embedding_model', args.embedding_model))
    con.commit()
    con.close()

    with docs_jsonl.open('w', encoding='utf-8') as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + '\n')
    style_clusters_path.write_text(json.dumps({'clusters': cluster_summary, 'vocab_size': len(vocab)}, ensure_ascii=False, indent=2), encoding='utf-8')

    llm = OpenAITextClient(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
        model=args.openai_model,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        read_timeout=args.openai_read_timeout,
        connect_timeout=args.openai_connect_timeout,
        retries=args.openai_retries,
    )
    try:
        caption_matrix = llm.embed_many(caption_embed_inputs)
        sticker_matrix = llm.embed_many(sticker_embed_inputs)
    finally:
        llm.close()
    np.save(caption_npy, caption_matrix.astype(np.float32))
    np.save(sticker_npy, sticker_matrix.astype(np.float32))
    _write_validation_report(built_rows=built_rows, output_path=validation_report_path, embedding_dimensions=args.embedding_dimensions, embedding_model=args.embedding_model)
    manifest_path.write_text(json.dumps({'sticker_ids': sticker_ids, 'dimensions': args.embedding_dimensions, 'embedding_model': args.embedding_model, 'build_stack': 'paddleocr+openai+tantivy+embeddings', 'schema_version': STICKER_SCHEMA_VERSION}, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"Indexed {len(sticker_ids)} stickers into {index_db} (new_ok={build_stats['ok']} skipped={build_stats['skipped']} failed={build_stats['failed']})")
    print(f'Wrote Tantivy docs JSONL to {docs_jsonl}')
    print(f'Wrote style clusters to {style_clusters_path}')
    print(f'Wrote embeddings to {caption_npy} and {sticker_npy}')
    print(f'Wrote validation report to {validation_report_path}')
    if build_stats['failed']:
        print(f'Logged per-sticker failures to {failures_log_path}')


if __name__ == '__main__':
    mp.freeze_support()
    main()
