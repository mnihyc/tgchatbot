#!/usr/bin/env bash
set -Eeuo pipefail
image=${1:?Pass the locally loaded image tag}
name=tgchatbot-smoke-$$
trap 'docker rm -f "$name" >/dev/null 2>&1 || true; docker volume rm "$name-data" "$name-tmp" >/dev/null 2>&1 || true' EXIT
docker volume create "$name-data" >/dev/null
docker volume create "$name-tmp" >/dev/null
docker run --rm --entrypoint sh -v "$name-data:/app/data" "$image" -c 'chown 12345:12345 /app/data'
docker run --rm -v "$name-data:/app/data" -v "$name-tmp:/tmp" "$image" python -c \
  'import os,pwd,av,numpy,PIL,tgchatbot.app; from pathlib import Path; assert os.getuid()==12345; assert pwd.getpwuid(os.getuid()).pw_dir=="/app/data/home"; assert os.stat("/tmp").st_uid==12345; Path("/tmp/writable").write_text("ok")'
docker run --rm --entrypoint sh -v "$name-data:/app/data" "$image" -c 'chown 0:0 /app/data'
docker run --rm -v "$name-data:/app/data" -v "$name-tmp:/tmp" "$image" python -c \
  'import os,pwd; from pathlib import Path; assert os.getuid()==0; assert pwd.getpwuid(0).pw_dir=="/app/data/home"; assert os.stat("/tmp").st_uid==0; Path("/tmp/root-writable").write_text("ok")'
# Prove the released maintenance stack loads and computes on CPU. Networking
# is disabled so this check cannot download OCR weights or call model APIs.
docker run --rm --network none "$image" sh -ec '
python - <<"PY"
import numpy as np
import paddle
from paddleocr import PaddleOCR
paddle.set_device("cpu")
value = paddle.to_tensor([[1, 2], [3, 4]], dtype="float32")
np.testing.assert_allclose(paddle.matmul(value, value).numpy(), [[7, 10], [15, 22]])
assert PaddleOCR is not None
PY
for tool in build_sticker_index rebuild_tantivy_index reset_sticker query_sticker_index show_style_clusters migrate_uid_json merge_auto_note_rows; do
    python "/app/scripts/$tool.py" --help >/dev/null
done
'
docker run --detach --name "$name" "$image" retriever
for attempt in $(seq 1 30); do
  if docker exec "$name" python -c \
    'import json,urllib.request; h=json.load(urllib.request.urlopen("http://127.0.0.1:4107/health",timeout=2)); assert h["status"]=="ok" and h["service"]=="tantivy_retriever" and h["schema_version"]=="human_sticker_v1"' >/dev/null 2>&1; then
    docker exec "$name" python -c \
      'import json,urllib.request; q=urllib.request.Request("http://127.0.0.1:4107/search",data=json.dumps({"caption_query_text":"hello","sticker_query_text":"hello"}).encode(),headers={"Content-Type":"application/json"}); assert json.load(urllib.request.urlopen(q,timeout=3))["hits"]==[]'
    docker exec -i --user tgchatbot "$name" python - <<'PY'
import json
from pathlib import Path
import subprocess
import time
import urllib.request

# Exercise the released Python CLI against the real indexer and HTTP server.
docs = Path('/tmp/fixture-docs.jsonl')
index = Path('/tmp/fixture-index')
command = ['python', '/app/scripts/rebuild_tantivy_index.py', '--docs-jsonl', str(docs), '--index-dir', str(index)]

def document(sticker_id, text):
    return json.dumps({'sticker_id': sticker_id, 'source_overlay_text_normalized': text,
                      'caption_semantic_text': text, 'sticker_semantic_text': text}) + '\n'

def serve():
    process = subprocess.Popen(['sticker-retriever', 'serve', '--index-dir', str(index), '--port', '4108'])
    for attempt in range(30):
        try:
            urllib.request.urlopen('http://127.0.0.1:4108/health', timeout=1).close()
            return process
        except OSError:
            time.sleep(0.1)
    process.terminate()
    process.wait(timeout=5)
    raise AssertionError('Fixture retriever did not start')

def search(text):
    request = urllib.request.Request('http://127.0.0.1:4108/search',
        data=json.dumps({'caption_query_text': text, 'sticker_query_text': text}).encode(),
        headers={'Content-Type': 'application/json'})
    return [hit['sticker_id'] for hit in json.load(urllib.request.urlopen(request, timeout=3))['hits']]

docs.write_text(document('fixture-old', 'oldonly'))
subprocess.run(command, check=True)
server = serve()
try:
    assert search('oldonly') == ['fixture-old']
finally:
    server.terminate()
    server.wait(timeout=5)

# Rebuild an existing index offline; old documents must disappear.
replacement = document('fixture-new', 'newonly')
docs.write_text(replacement)
subprocess.run(command, check=True)
server = serve()
try:
    assert search('newonly') == ['fixture-new']
    assert search('oldonly') == []
finally:
    server.terminate()
    server.wait(timeout=5)

# A failed offline rebuild must preserve an index that a fresh server can open.
docs.write_text('{malformed JSON\n')
failed = subprocess.run(command, capture_output=True, text=True)
assert failed.returncode != 0, 'Malformed documents unexpectedly rebuilt the index'
server = serve()
try:
    assert search('newonly') == ['fixture-new'], 'Failed rebuild damaged the working index'
    assert search('oldonly') == []
    assert not list(index.parent.glob('.fixture-index.*')), 'Rebuild left staging/backup debris'
finally:
    server.terminate()
    server.wait(timeout=5)
    docs.write_text(replacement)
PY
    # Restart with changed source documents: an already-created index must be
    # reused, never silently rebuilt during normal container startup.
    before=$(docker exec "$name" sha256sum /app/data/tantivy_index/meta.json)
    docker exec "$name" cp /tmp/fixture-docs.jsonl /app/data/tantivy_docs.jsonl
    docker restart "$name" >/dev/null
    after=$(docker exec "$name" sha256sum /app/data/tantivy_index/meta.json)
    [[ $before == "$after" ]]
    exit 0
  fi
  sleep 1
done
docker logs "$name"
exit 1
