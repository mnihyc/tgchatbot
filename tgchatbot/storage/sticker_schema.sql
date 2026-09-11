-- Sticker catalog revisions are independent of private chat/agent generations.
CREATE TABLE IF NOT EXISTS sticker_catalog_revisions (
    id text PRIMARY KEY,
    parent_id text REFERENCES sticker_catalog_revisions(id),
    source_root text NOT NULL,
    recipe jsonb NOT NULL,
    state text NOT NULL DEFAULT 'staging' CHECK (state IN ('staging','active','superseded')),
    created_at timestamptz NOT NULL DEFAULT now(),
    activated_at timestamptz
);
CREATE TABLE IF NOT EXISTS sticker_catalog_head (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    revision_id text REFERENCES sticker_catalog_revisions(id)
);
INSERT INTO sticker_catalog_head(singleton) VALUES(true) ON CONFLICT DO NOTHING;
-- Vector bytes are content-addressed separately so revision history shares them.
CREATE TABLE IF NOT EXISTS sticker_catalog_vectors (
    id text PRIMARY KEY,
    dimensions integer NOT NULL,
    payload bytea NOT NULL
);
CREATE TABLE IF NOT EXISTS sticker_catalog_items (
    revision_id text NOT NULL REFERENCES sticker_catalog_revisions(id),
    asset_id text NOT NULL,
    content_hash text NOT NULL,
    aliases jsonb NOT NULL,
    media jsonb NOT NULL DEFAULT '{}',
    generated_card jsonb,
    corrections jsonb NOT NULL DEFAULT '{}',
    card jsonb,
    provenance jsonb NOT NULL DEFAULT '{}',
    dimensions integer NOT NULL DEFAULT 0,
    image_vector_id text REFERENCES sticker_catalog_vectors(id),
    reading_vectors_id text REFERENCES sticker_catalog_vectors(id),
    state text NOT NULL DEFAULT 'pending' CHECK (state IN ('pending','ready','failed')),
    error text,
    PRIMARY KEY (revision_id,asset_id)
);
