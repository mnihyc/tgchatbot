"""Versioned sticker catalog. Active readers never observe a partially built revision."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.resources import files
import hashlib
import struct
import uuid
from typing import Any

import numpy as np
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


ITEM_UPSERT = '''INSERT INTO sticker_catalog_items
                    (revision_id,asset_id,content_hash,aliases,media,generated_card,corrections,card,provenance,
                     dimensions,image_vector_id,reading_vectors_id,state,error)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(revision_id,asset_id) DO UPDATE SET aliases=EXCLUDED.aliases,media=EXCLUDED.media,
                    generated_card=EXCLUDED.generated_card,corrections=EXCLUDED.corrections,card=EXCLUDED.card,
                    provenance=EXCLUDED.provenance,dimensions=EXCLUDED.dimensions,image_vector_id=EXCLUDED.image_vector_id,
                    reading_vectors_id=EXCLUDED.reading_vectors_id,state=EXCLUDED.state,error=EXCLUDED.error'''

ITEM_SELECT = """SELECT item.*, image.payload AS image_vector, readings.payload AS reading_vectors
    FROM sticker_catalog_items item
    LEFT JOIN sticker_catalog_vectors image ON image.id=item.image_vector_id
    LEFT JOIN sticker_catalog_vectors readings ON readings.id=item.reading_vectors_id"""

@dataclass(frozen=True)
class CatalogAlias:
    path: str
    pack: str


@dataclass(frozen=True)
class CatalogAsset:
    asset_id: str
    content_hash: str
    aliases: tuple[CatalogAlias, ...]
    media: dict
    generated_card: dict | None
    corrections: dict
    card: dict | None
    provenance: dict
    image_vector: np.ndarray | None
    reading_vectors: np.ndarray | None
    state: str = 'ready'
    error: str | None = None

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(self.corrections.get('family_ids', ()))

    @property
    def style_tags(self) -> tuple[str, ...]:
        return tuple(self.corrections.get('style_tags', ()))


@dataclass(frozen=True)
class CatalogSnapshot:
    revision_id: str | None
    recipe: dict
    source_root: str
    assets: tuple[CatalogAsset, ...]


_UNSET = object()


class CatalogConflict(RuntimeError):
    pass


class StickerCatalogStore:
    def __init__(self, postgres_store):
        self.store = postgres_store

    async def initialize(self):
        async with self.store.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',
                                   ('sticker-catalog-schema:' + self.store.schema,))
                await conn.execute(files('tgchatbot.storage').joinpath('sticker_schema.sql').read_text())

    async def active_revision_id(self) -> str | None:
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute('SELECT revision_id FROM sticker_catalog_head WHERE singleton')).fetchone()
        return row['revision_id'] if row else None

    @staticmethod
    def _asset(row: dict) -> CatalogAsset:
        dimensions = row['dimensions']
        def vector(value, matrix=False):
            if value is None:
                return None
            result = np.frombuffer(value, dtype='<f4').copy()
            if matrix:
                result = result.reshape((-1, dimensions))
            result.setflags(write=False)
            return result
        return CatalogAsset(row['asset_id'], row['content_hash'],
            tuple(CatalogAlias(**alias) for alias in row['aliases']), row['media'], row['generated_card'],
            row['corrections'], row['card'], row['provenance'], vector(row['image_vector']),
            vector(row['reading_vectors'], True), row['state'], row['error'])

    async def load_snapshot(self, revision_id: str | None = None) -> CatalogSnapshot:
        # Revisions are immutable after activation. Capture head and its contents
        # in one snapshot so publication during this read cannot mix revisions.
        async with self.store.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
                if revision_id is None:
                    row = await (await conn.execute('SELECT revision_id FROM sticker_catalog_head WHERE singleton')).fetchone()
                    revision_id = row['revision_id'] if row else None
                if revision_id is None:
                    return CatalogSnapshot(None, {}, '', ())
                revision = await (await conn.execute('SELECT * FROM sticker_catalog_revisions WHERE id=%s', (revision_id,))).fetchone()
                if revision is None:
                    raise KeyError(revision_id)
                rows = await (await conn.execute(ITEM_SELECT + ' WHERE item.revision_id=%s ORDER BY item.asset_id', (revision_id,))).fetchall()
        return CatalogSnapshot(revision_id, revision['recipe'], revision['source_root'], tuple(self._asset(row) for row in rows))

    async def get_asset(self, asset_id: str, revision_id: str | None = None) -> CatalogAsset | None:
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute(ITEM_SELECT + ''' WHERE item.asset_id=%s AND
                item.revision_id=COALESCE(%s,(SELECT revision_id FROM sticker_catalog_head WHERE singleton))''',
                (asset_id, revision_id))).fetchone()
        return self._asset(row) if row else None

    async def begin_revision(self, *, source_root: str, recipe: dict, assets: list[CatalogAsset] | None = None, dimensions: int = 0, expected_parent: str | None | object = _UNSET) -> str:
        revision_id = uuid.uuid4().hex
        async with self.store.pool.connection() as conn:
            async with conn.transaction():
                head = await (await conn.execute('SELECT revision_id FROM sticker_catalog_head WHERE singleton FOR UPDATE')).fetchone()
                parent = head['revision_id']
                if expected_parent is not _UNSET and parent != expected_parent:
                    raise CatalogConflict('Active catalog changed during inventory; retry from the current catalog')
                await conn.execute('INSERT INTO sticker_catalog_revisions(id,parent_id,source_root,recipe) VALUES(%s,%s,%s,%s)',
                                   (revision_id, parent, source_root, Jsonb(recipe)))
                if parent and assets is None:
                    await conn.execute('''INSERT INTO sticker_catalog_items
                        SELECT %s,asset_id,content_hash,aliases,media,generated_card,corrections,card,provenance,
                               dimensions,image_vector_id,reading_vectors_id,state,error
                        FROM sticker_catalog_items WHERE revision_id=%s''', (revision_id, parent))
                if assets is not None:
                    for asset in assets:
                        await conn.execute(ITEM_UPSERT, (revision_id, asset.asset_id, asset.content_hash,
                            Jsonb([{'path': a.path, 'pack': a.pack} for a in asset.aliases]), Jsonb(asset.media),
                            Jsonb(asset.generated_card), Jsonb(asset.corrections), Jsonb(asset.card),
                            Jsonb(asset.provenance), dimensions, await self._save_vector(conn, asset.image_vector, dimensions),
                            await self._save_vector(conn, asset.reading_vectors, dimensions), asset.state, asset.error))
        return revision_id

    @asynccontextmanager
    async def build_lease(self, revision_id: str):
        # A dedicated maintenance connection owns the session lock. Do not occupy
        # the only application-pool slot when an operator configures pool size 1.
        async with await AsyncConnection.connect(self.store.dsn, autocommit=True, row_factory=dict_row) as conn:
            locked = await (await conn.execute('SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS acquired',
                                               ('sticker-build:' + revision_id,))).fetchone()
            if not locked['acquired']:
                raise CatalogConflict('This catalog revision is already being built')
            try:
                # Reject already obsolete work before buying annotation/embedding
                # requests. Activation still checks again to resolve publication races.
                async with self.store.pool.connection() as state_conn:
                    revision = await (await state_conn.execute('''SELECT r.state,r.parent_id,h.revision_id
                        FROM sticker_catalog_revisions r CROSS JOIN sticker_catalog_head h
                        WHERE r.id=%s AND h.singleton''', (revision_id,))).fetchone()
                if not revision or revision['state'] != 'staging':
                    raise CatalogConflict('Only a staging revision can be built')
                if revision['parent_id'] != revision['revision_id']:
                    raise CatalogConflict('Active catalog changed; start a new revision from the current catalog')
                yield
            finally:
                await conn.execute('SELECT pg_advisory_unlock(hashtextextended(%s,0))', ('sticker-build:' + revision_id,))

    async def stage_asset(self, revision_id: str, *, asset_id: str, content_hash: str,
                          aliases: list[dict], media: dict, generated_card: dict | None,
                          corrections: dict, card: dict | None, provenance: dict,
                          dimensions: int = 0, image_vector=None, reading_vectors=None,
                          state: str = 'pending', error: str | None = None):
        async with self.store.pool.connection() as conn:
            async with conn.transaction():
                revision = await (await conn.execute('SELECT state FROM sticker_catalog_revisions WHERE id=%s FOR UPDATE', (revision_id,))).fetchone()
                if not revision or revision['state'] != 'staging':
                    raise CatalogConflict('Only a staging revision can be changed')
                await conn.execute(ITEM_UPSERT, (revision_id, asset_id, content_hash, Jsonb(aliases), Jsonb(media),
                    Jsonb(generated_card), Jsonb(corrections), Jsonb(card), Jsonb(provenance), dimensions,
                    await self._save_vector(conn, image_vector, dimensions),
                    await self._save_vector(conn, reading_vectors, dimensions), state, error))

    @staticmethod
    async def _save_vector(conn, vector, dimensions):
        if vector is None:
            return None
        payload = np.asarray(vector, dtype='<f4').tobytes()
        identity = hashlib.sha256(struct.pack('>QQ', dimensions, len(payload)) + payload).hexdigest()
        await conn.execute('INSERT INTO sticker_catalog_vectors(id,dimensions,payload) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING',
                           (identity, dimensions, payload))
        return identity

    async def activate(self, revision_id: str):
        async with self.store.pool.connection() as conn:
            async with conn.transaction():
                head = await (await conn.execute('SELECT revision_id FROM sticker_catalog_head WHERE singleton FOR UPDATE')).fetchone()
                revision = await (await conn.execute('SELECT * FROM sticker_catalog_revisions WHERE id=%s FOR UPDATE', (revision_id,))).fetchone()
                if not revision or revision['state'] != 'staging':
                    raise CatalogConflict('Only a staging revision can be activated')
                if revision['parent_id'] != head['revision_id']:
                    raise CatalogConflict('Active catalog changed; start a new revision from the current catalog')
                pending = await (await conn.execute("SELECT count(*) AS count FROM sticker_catalog_items WHERE revision_id=%s AND state<>'ready'", (revision_id,))).fetchone()
                if pending['count']:
                    raise CatalogConflict(f"Revision still has {pending['count']} unfinished assets")
                recipe = revision['recipe']
                if recipe.get('embedding_space_id'):
                    # Publication is the consistency boundary for the two vector
                    # channels and their explicitly selected embedding space.
                    incomplete = await (await conn.execute("""SELECT item.asset_id FROM sticker_catalog_items item
                        LEFT JOIN sticker_catalog_vectors image ON image.id=item.image_vector_id
                        LEFT JOIN sticker_catalog_vectors readings ON readings.id=item.reading_vectors_id
                        WHERE item.revision_id=%s AND (item.card IS NULL OR item.card='null'::jsonb OR
                        item.provenance->>'embedding_space_id' IS DISTINCT FROM %s OR item.dimensions<>%s OR
                        readings.payload IS NULL OR octet_length(readings.payload) <> 4*item.dimensions*jsonb_array_length(item.card->'readings') OR
                        ((%s OR (%s AND NULLIF(regexp_replace(concat_ws('',item.card->>'appearance',
                            item.card->>'action',item.card->>'caption'),'[[:space:]]','','g'),'') IS NOT NULL))
                        AND (image.payload IS NULL OR octet_length(image.payload)<>4*item.dimensions))) LIMIT 1""",
                        (revision_id, recipe['embedding_space_id'], recipe['embedding_space']['dimensions'],
                         bool(recipe.get('image_embeddings')), recipe.get('visual_embedding_source') == 'description'))).fetchone()
                    if incomplete:
                        raise CatalogConflict('Revision contains incomplete or incompatible vectors: ' + incomplete['asset_id'])
                await conn.execute("UPDATE sticker_catalog_revisions SET state='superseded' WHERE id=%s", (head['revision_id'],))
                await conn.execute("UPDATE sticker_catalog_revisions SET state='active',activated_at=now() WHERE id=%s", (revision_id,))
                await conn.execute('UPDATE sticker_catalog_head SET revision_id=%s WHERE singleton', (revision_id,))
