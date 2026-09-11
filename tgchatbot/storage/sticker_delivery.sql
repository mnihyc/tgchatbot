CREATE TABLE IF NOT EXISTS sticker_deliveries (
    operation_id text PRIMARY KEY,
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    context_id bigint NOT NULL,
    revision bigint NOT NULL,
    sticker_id text NOT NULL,
    timing text NOT NULL,
    status text NOT NULL CHECK (status IN ('queued','sending','sent','failed','unknown')),
    telegram_message_id bigint,
    error text,
    metadata jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS sticker_deliveries_recent
    ON sticker_deliveries(session_id,generation,finished_at DESC)
    WHERE status='sent';
CREATE INDEX IF NOT EXISTS sticker_deliveries_unresolved
    ON sticker_deliveries(session_id,generation,created_at DESC)
    WHERE status IN ('sending','unknown');
