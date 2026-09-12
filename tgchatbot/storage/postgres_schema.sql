-- Originals are immutable revisions. Context, memory, and indexing are projections.
CREATE TABLE IF NOT EXISTS schema_version (
    version integer PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id text PRIMARY KEY,
    settings jsonb NOT NULL DEFAULT '{}',
    generation bigint NOT NULL DEFAULT 1,
    context_id bigint NOT NULL DEFAULT 1,
    revision bigint NOT NULL DEFAULT 1,
    context_version bigint NOT NULL DEFAULT 0,
    compaction_version bigint NOT NULL DEFAULT 0,
    profile_refresh_version bigint NOT NULL DEFAULT 0,
    sticker_persona jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- Compressed image evidence, deduplicated per conversation. Canonical message
-- revisions own these references through prompt retirement and resets, including
-- prior revisions/generations retained for audit. Runtime caches are disposable.
-- Physical audit deletion, not working-context retirement, governs byte removal.
CREATE TABLE IF NOT EXISTS message_previews (
    session_id text NOT NULL REFERENCES sessions(session_id),
    reference text NOT NULL,
    payload bytea NOT NULL,
    PRIMARY KEY (session_id, reference)
);

CREATE TABLE IF NOT EXISTS provider_token_calibration (
    provider text NOT NULL,
    model text NOT NULL,
    history_mode text NOT NULL,
    multiplier double precision NOT NULL,
    PRIMARY KEY (provider, model, history_mode)
);

-- A full reset retires agent personality/settings without erasing its audit state.
-- Ordinary retrieval never reads this operator-only snapshot table.
CREATE TABLE IF NOT EXISTS agent_generations (
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    context_id bigint NOT NULL,
    scope_revision bigint NOT NULL,
    settings jsonb NOT NULL,
    sticker_persona jsonb,
    retired_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, generation)
);

CREATE TABLE IF NOT EXISTS messages (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    context_id bigint NOT NULL,
    source text NOT NULL DEFAULT 'unknown',
    source_chat_id text,
    source_message_id text,
    actor_id text,
    actor_kind text NOT NULL DEFAULT 'unknown',
    actor_name text,
    topic_id text,
    reply_to_source_id text,
    role text NOT NULL,
    source_revision integer NOT NULL DEFAULT 1,
    hidden boolean NOT NULL DEFAULT false,
    deleted boolean NOT NULL DEFAULT false,
    compacted_by_block_id bigint,
    presentation jsonb,
    sent_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, generation, source, source_chat_id, source_message_id)
);
CREATE INDEX IF NOT EXISTS messages_context ON messages (session_id, generation, context_id, id DESC)
    WHERE NOT hidden AND NOT deleted;
CREATE INDEX IF NOT EXISTS messages_actor_time ON messages (session_id, generation, actor_id, sent_at, id)
    WHERE NOT hidden AND NOT deleted;
CREATE INDEX IF NOT EXISTS messages_time ON messages (session_id, generation, sent_at, id)
    WHERE NOT hidden AND NOT deleted;
CREATE INDEX IF NOT EXISTS messages_topic_time ON messages (session_id, generation, topic_id, sent_at, id)
    WHERE NOT hidden AND NOT deleted;

-- Extra Telegram answer chunks map to one canonical body. The first chunk
-- uses messages' existing source identity index, avoiding a row for single-part replies.
CREATE TABLE IF NOT EXISTS message_source_aliases (
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    source text NOT NULL,
    source_chat_id text NOT NULL,
    source_message_id text NOT NULL,
    message_id bigint NOT NULL REFERENCES messages(id),
    PRIMARY KEY (session_id, generation, source, source_chat_id, source_message_id)
);

CREATE TABLE IF NOT EXISTS message_revisions (
    message_id bigint NOT NULL REFERENCES messages(id),
    revision integer NOT NULL,
    body text NOT NULL,
    parts jsonb NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}',
    estimated_tokens integer NOT NULL,
    fingerprint text NOT NULL,
    edited_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (message_id, revision)
);
-- Only the current searchable revision is indexed, so editing does not leave old
-- assertions in the ordinary lexical results. The original body is not copied.
CREATE TABLE IF NOT EXISTS message_search (
    message_id bigint PRIMARY KEY REFERENCES messages(id),
    lexemes tsvector NOT NULL
);
CREATE INDEX IF NOT EXISTS message_search_lexical ON message_search USING gin (lexemes);

CREATE TABLE IF NOT EXISTS memory_blocks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    context_id bigint NOT NULL,
    sequence_no bigint NOT NULL,
    summary_text text NOT NULL,
    estimated_tokens integer NOT NULL,
    source_ids bigint[] NOT NULL,
    source_revisions jsonb NOT NULL,
    details jsonb NOT NULL DEFAULT '{}',
    valid boolean NOT NULL DEFAULT true,
    superseded_by bigint,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memory_blocks_context ON memory_blocks (session_id, generation, context_id, sequence_no)
    WHERE valid;
CREATE INDEX IF NOT EXISTS memory_blocks_sources ON memory_blocks USING gin (source_ids) WHERE valid;

CREATE TABLE IF NOT EXISTS excerpts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    source_ids bigint[] NOT NULL,
    source_revisions jsonb NOT NULL,
    spans jsonb NOT NULL DEFAULT '[]',
    fingerprint text NOT NULL,
    model text NOT NULL DEFAULT '',
    -- Inline vectors avoid hundreds of thousands of TOAST lookups during exact
    -- scoped search. All 1536 dimensions remain; no approximate index is needed.
    embedding halfvec(1536) STORAGE PLAIN,
    valid boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, generation, fingerprint, model)
) WITH (toast_tuple_target=4096);
-- A 3 KiB inline vector already exceeds the default 2 KiB TOAST target. A
-- 4 KiB target keeps its ordinary identity/source metadata inline too, while
-- allowing wide source lists to spill. Normal excerpts still fit two per page.
CREATE INDEX IF NOT EXISTS excerpts_scope ON excerpts (session_id, generation, id) WHERE valid;
-- Background excerpt writes update this index immediately. Pending GIN lists
-- otherwise get rescanned by every source probe in a foreground actor lookup.
CREATE INDEX IF NOT EXISTS excerpts_sources ON excerpts USING gin (source_ids)
    WITH (fastupdate=off) WHERE valid;
-- Invalidated vectors also need bounded retirement. Cleared rows leave this index.
CREATE INDEX IF NOT EXISTS excerpts_retirement ON excerpts (session_id, generation, id)
    WHERE valid OR embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS excerpt_tails (
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    topic_id text NOT NULL DEFAULT '',
    source_ids bigint[] NOT NULL,
    source_revisions jsonb NOT NULL,
    spans jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, generation, topic_id)
);
CREATE INDEX IF NOT EXISTS excerpt_tails_sources ON excerpt_tails USING gin (source_ids);

CREATE TABLE IF NOT EXISTS profile_facts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    subject_actor_id text NOT NULL,
    asserted_by text NOT NULL,
    claim text NOT NULL,
    claim_key text,
    fingerprint text NOT NULL,
    supersedes bigint REFERENCES profile_facts(id),
    kind text NOT NULL CHECK (kind IN ('explicit', 'inferred')),
    status text NOT NULL DEFAULT 'active',
    valid_from timestamptz,
    valid_to timestamptz,
    original_valid_to timestamptz,
    retired_at timestamptz,
    retirement_sources bigint[],
    source_ids bigint[] NOT NULL,
    source_revisions jsonb NOT NULL,
    valid boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from),
    UNIQUE (session_id, generation, fingerprint)
);
CREATE INDEX IF NOT EXISTS profile_facts_actor ON profile_facts (session_id, generation, subject_actor_id, id DESC)
    WHERE valid;
CREATE INDEX IF NOT EXISTS profile_facts_sources ON profile_facts USING gin (source_ids) WHERE valid;
CREATE INDEX IF NOT EXISTS profile_facts_corrections ON profile_facts (supersedes) WHERE valid AND supersedes IS NOT NULL;

-- Original text stays in message_revisions. Only the unprocessed source spans
-- live here; a consumed row prevents indexing/rebuild from learning it twice.
CREATE TABLE IF NOT EXISTS profile_inputs (
    message_id bigint PRIMARY KEY REFERENCES messages(id),
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    source_revision integer NOT NULL,
    actor_id text NOT NULL,
    spans jsonb NOT NULL,
    pending_bytes bigint NOT NULL
);
CREATE INDEX IF NOT EXISTS profile_inputs_pending ON profile_inputs (session_id, generation, message_id)
    WHERE pending_bytes > 0;

CREATE INDEX IF NOT EXISTS profile_facts_retirement_sources ON profile_facts USING gin (retirement_sources)
    WHERE retired_at IS NOT NULL;
CREATE TABLE IF NOT EXISTS profile_current (
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    actor_id text NOT NULL,
    fact_ids bigint[] NOT NULL DEFAULT '{}',
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(session_id,generation,actor_id)
);
CREATE TABLE IF NOT EXISTS profile_patches (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    source_ids bigint[] NOT NULL,
    source_revisions jsonb NOT NULL,
    patch jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS profile_patches_session ON profile_patches(session_id,id);

CREATE TABLE IF NOT EXISTS jobs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id text NOT NULL REFERENCES sessions(session_id),
    generation bigint NOT NULL,
    context_id bigint NOT NULL,
    scope_revision bigint NOT NULL,
    kind text NOT NULL,
    policy text NOT NULL CHECK (policy IN ('memory', 'context')),
    source_ids bigint[] NOT NULL DEFAULT '{}',
    source_revisions jsonb NOT NULL DEFAULT '{}',
    payload jsonb NOT NULL DEFAULT '{}',
    dedupe_key text,
    status text NOT NULL DEFAULT 'pending',
    attempts integer NOT NULL DEFAULT 0,
    lease_token text,
    lease_until timestamptz,
    available_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    error text,
    UNIQUE (session_id, generation, kind, dedupe_key)
);
CREATE INDEX IF NOT EXISTS jobs_pending ON jobs (available_at, id) WHERE status IN ('pending', 'running');
CREATE INDEX IF NOT EXISTS jobs_sources ON jobs USING gin (source_ids) WHERE status IN ('pending', 'running');

INSERT INTO schema_version (version) VALUES (3) ON CONFLICT DO NOTHING;
