# syntax=docker/dockerfile:1
FROM rust:1.89-bookworm AS retriever-build
WORKDIR /build
COPY retriever/Cargo.toml retriever/Cargo.lock ./
COPY retriever/src ./src
RUN cargo build --locked --release

FROM ghcr.io/astral-sh/uv:0.11.16 AS uv
FROM python:3.12-slim-bookworm AS runtime
ARG RELEASE_TAG=development
ARG RELEASE_COMMIT=unknown
LABEL org.opencontainers.image.title="tgchatbot" \
      org.opencontainers.image.version=$RELEASE_TAG \
      org.opencontainers.image.revision=$RELEASE_COMMIT
# PyAV/Pillow use locked binary wheels; SSH powers the existing remote tools.
# util-linux supplies setpriv for dropping to the host data owner's identity.
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client ca-certificates util-linux \
    && apt-get clean \
    && groupadd --gid 1000 tgchatbot \
    && useradd --uid 1000 --gid 1000 --home-dir /app/data/home --no-create-home tgchatbot
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    APP_DATA_DIR=/app/data HOME=/app/data/home
COPY pyproject.toml uv.lock ./
COPY tgchatbot ./tgchatbot
RUN uv sync --frozen --no-dev --no-editable
COPY --from=retriever-build /build/target/release/sticker-retriever /usr/local/bin/sticker-retriever
COPY scripts ./scripts
COPY deploy/entrypoint.sh /usr/local/bin/tgchatbot-entrypoint
ENTRYPOINT ["/usr/local/bin/tgchatbot-entrypoint"]
CMD ["python", "-m", "tgchatbot.app"]
