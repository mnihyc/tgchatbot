# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends openssh-client ca-certificates util-linux \
    && apt-get clean \
    && groupadd --gid 1000 tgchatbot \
    && useradd --uid 1000 --gid 1000 --home-dir /app/data/home --no-create-home tgchatbot \
    && sed -i '/^root:/s#:/root:#:/app/data/home:#' /etc/passwd \
    && install -d -o 1000 -g 1000 /app/data

COPY build/runtime-requirements.txt /opt/tgchatbot-requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: -r /opt/tgchatbot-requirements.txt

ARG RELEASE_TAG=development
ARG RELEASE_COMMIT=unknown
LABEL org.opencontainers.image.title="tgchatbot" \
      org.opencontainers.image.source="https://github.com/mnihyc/tgchatbot" \
      org.opencontainers.image.version=$RELEASE_TAG \
      org.opencontainers.image.revision=$RELEASE_COMMIT
COPY build/tgchatbot-*.whl /opt/tgchatbot-wheel/
RUN python -m pip install --no-cache-dir --no-deps --no-index /opt/tgchatbot-wheel/*.whl
COPY --chmod=755 deploy/entrypoint.sh /usr/local/bin/tgchatbot-entrypoint
COPY deploy/configure_database.py /usr/local/lib/tgchatbot-deploy-configure.py

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    APP_DATA_DIR=/app/data APP_TEMP_DIR=/tmp HOME=/app/data/home
WORKDIR /app
ENTRYPOINT ["/usr/local/bin/tgchatbot-entrypoint"]
CMD ["python", "-m", "tgchatbot.app"]
