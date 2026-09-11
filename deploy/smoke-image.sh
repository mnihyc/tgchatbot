#!/usr/bin/env bash
set -Eeuo pipefail
image=${1:?Pass the locally loaded image tag}
name=tgchatbot-smoke-$$
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
docker run --rm -e TGCHATBOT_UID=12345 -e TGCHATBOT_GID=12345 "$image" python -c \
  'import os,pwd,av,numpy,PIL,tgchatbot.app; assert os.getuid()==12345; assert pwd.getpwuid(os.getuid()).pw_dir=="/app/data/home"'
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

# Search a real indexed fixture; this catches tokenizer/schema/linking defects
# that an empty-index health probe cannot find.
docs = Path('/tmp/fixture-docs.jsonl')
docs.write_text(json.dumps({'sticker_id': 'fixture-wave', 'source_overlay_text_normalized': 'hello',
                           'caption_semantic_text': 'hello wave', 'sticker_semantic_text': 'friendly greeting'}) + '\n')
subprocess.run(['sticker-retriever', 'build', '--docs-jsonl', str(docs), '--index-dir', '/tmp/fixture-index'], check=True)
server = subprocess.Popen(['sticker-retriever', 'serve', '--index-dir', '/tmp/fixture-index', '--port', '4108'])
try:
    for attempt in range(30):
        try:
            urllib.request.urlopen('http://127.0.0.1:4108/health', timeout=1).close()
            break
        except OSError:
            time.sleep(0.1)
    request = urllib.request.Request('http://127.0.0.1:4108/search',
        data=json.dumps({'caption_query_text': 'hello', 'sticker_query_text': 'friendly'}).encode(),
        headers={'Content-Type': 'application/json'})
    result = json.load(urllib.request.urlopen(request, timeout=3))
    assert any(hit['sticker_id'] == 'fixture-wave' for hit in result['hits']), result
finally:
    server.terminate()
    server.wait(timeout=5)
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
