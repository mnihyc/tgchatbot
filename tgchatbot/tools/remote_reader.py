"""File parsing program executed by the workspace's Python, never by the bot.

The shared image encoder is prepended by RemoteWorkspaceClient. Only selected
text/pixels leave the workspace; originals are never modified.
"""
import base64
from contextlib import closing
import json
import math
from pathlib import Path
import sys


def read_document(request):
    path = Path(request['path']).resolve()
    if not path.is_relative_to(Path(request['root']).resolve()):
        raise ValueError('File is outside this workspace')
    if not path.is_file():
        raise ValueError('File is unavailable')
    fmt = request['file_format']
    start, end = request.get('start'), request.get('end')
    if fmt not in {'text', 'image', 'pdf'}:
        raise ValueError('This provider cannot receive recording tool results')
    if fmt == 'image' and (start is not None or end is not None):
        raise ValueError('Line/page selection applies only to text or PDF')
    if any(value is not None and (type(value) is not int or value < 1) for value in (start, end)):
        raise ValueError('Line/page numbers must be positive integers')
    start = start or 1
    if end is not None and end < start:
        raise ValueError('Selection end precedes its start')
    parts, used_tokens, used_bytes, used_images = [], 0, 0, 0
    limits = request['limits']
    details = {}

    def append(*new_parts, stop_when_full=False):
        nonlocal used_tokens, used_bytes, used_images
        tokens, size, images = used_tokens, used_bytes, used_images
        for part in new_parts:
            text = part.get('text', '')
            image = part['kind'] == 'image'
            tokens += limits['part_overhead'] + (
                limits['image_tokens'] if image else math.ceil(len(text) / limits['chars_per_token']))
            tokens += math.ceil(len(part.get('filename', '')) / limits['chars_per_token'])
            images += int(image)
            size += len(json.dumps(part, ensure_ascii=False).encode('utf-8'))
        if (tokens > limits['tokens'] or size > limits['bytes'] or
                (limits['images'] is not None and images > limits['images'])):
            if stop_when_full and parts:
                return False
            if stop_when_full:
                raise ValueError('No complete PDF page fits this call\'s remaining allowance')
            raise ValueError('Selection exceeds this read allowance; choose fewer lines or pages')
        used_tokens, used_bytes, used_images = tokens, size, images
        parts.extend(new_parts)
        return True

    def image_part(frame, label):
        max_pixels = limits['max_image_pixels']
        if max_pixels > 0 and frame.width * frame.height > max_pixels:
            ratio = math.sqrt(max_pixels / (frame.width * frame.height))
            frame.thumbnail((max(1, int(frame.width * ratio)), max(1, int(frame.height * ratio))))
        mime, data = compress_frame(frame, limits['bytes'])
        return {'kind': 'image', 'mime_type': mime, 'data_b64': base64.b64encode(data).decode(),
                'filename': label, 'text': label}

    if fmt == 'text':
        with path.open('rb') as source:
            if source.read(5) == b'%PDF-':
                raise ValueError('This file is a PDF; use file_format="pdf" with start/end page numbers')
        selected = []
        characters = 0
        last = 0
        allowance = max(1, int(limits['tokens'] * limits['chars_per_token']))
        try:
            with path.open(encoding='utf-8-sig') as stream:
                while end is None or last < end:
                    line = stream.readline(allowance + 1)
                    if not line:
                        break
                    last += 1
                    if last < start:
                        while line and not line.endswith('\n'):
                            line = stream.readline(allowance + 1)
                        continue
                    characters += len(line)
                    if characters > allowance:
                        raise ValueError('Selection exceeds this read allowance; choose fewer lines')
                    selected.append(line)
        except UnicodeDecodeError as exc:
            raise ValueError('This file is not UTF-8 text; choose its source file_format') from exc
        empty_whole_file = last == 0 and request.get('start') is None and end is None
        if not empty_whole_file and (start > last or (end is not None and last < end)):
            raise ValueError('Requested lines are outside the file')
        append({'kind': 'text', 'text': ''.join(selected)})
        selection = {'start': start, 'end': start + len(selected) - 1, 'unit': 'lines'}
    elif fmt == 'image':
        from PIL import Image, ImageOps
        with Image.open(path) as original:
            frames = getattr(original, 'n_frames', 1)
            with ImageOps.exif_transpose(original) as frame:
                append(image_part(frame, path.name))
        selection = {'frame': 1, 'total_frames': frames} if frames > 1 else None
    else:
        import pypdfium2 as pdfium
        # Each reader is a separate remote process: PDFium is never called
        # concurrently by threads in the same process.
        with pdfium.PdfDocument(path) as pdf:
            total_pages = len(pdf)
            details['total_pages'] = total_pages
            if start > total_pages:
                raise ValueError('Requested pages are outside the PDF')
            requested_end = min(total_pages, end) if end is not None else total_pages
            end = min(requested_end, start + limits['pdf_max_pages'] - 1)
            if limits['images'] is not None:
                end = min(end, start + limits['images'] - 1)
            if end < start:
                raise ValueError('No PDF page fits the current image allowance')
            last_page = start - 1
            for index in range(start - 1, end):
                with closing(pdf[index]) as page:
                    label = f'{path.name}, page {index + 1}'
                    with closing(page.get_textpage()) as textpage:
                        text = textpage.get_text_range()
                    text_part = {'kind': 'text', 'text': label + ('\n' + text if text else '')}
                    scale = limits['pdf_scale']
                    if scale <= 0:
                        raise ValueError('READ_DOC_PDF_SCALE must be positive')
                    max_pixels = limits['max_image_pixels']
                    if max_pixels > 0:
                        scale = min(scale, math.sqrt(max_pixels / (page.get_width() * page.get_height())))
                    with closing(page.render(scale=scale)) as bitmap:
                        with bitmap.to_pil() as frame:
                            picture = image_part(frame, label)
                    # Text and image describe one page; admit or omit them together.
                    if not append(text_part, picture, stop_when_full=True):
                        break
                    last_page = index + 1
            selection = {'start': start, 'end': last_page, 'unit': 'pages'}
            if last_page < requested_end:
                details['note'] = f'Additional pages were not included; continue with start={last_page + 1}.'
    return {'ok': True, 'file_format': fmt, 'selection': selection, **details, 'parts': parts}


if __name__ == '__main__':
    try:
        result = read_document(json.loads(sys.argv[1]))
    except ModuleNotFoundError as exc:
        result = {'ok': False, 'error': f'Remote Python is missing {exc.name}; configure the remote reader dependencies'}
    except Exception as exc:
        result = {'ok': False, 'error': str(exc)}
    print(json.dumps(result, ensure_ascii=False))
