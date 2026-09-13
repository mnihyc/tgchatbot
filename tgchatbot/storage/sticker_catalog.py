"""Versioned sticker catalog. Active readers never observe a partially built revision."""
from __future__ import annotations

from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from importlib.resources import files
import hashlib
import struct
import uuid
from typing import Any

import numpy as np
from psycopg import AsyncConnection, AsyncServerCursor
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
    pack_descriptions: dict[str, str] = field(default_factory=dict)


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
        return CatalogSnapshot(revision_id, revision['recipe'], revision['source_root'],
                               tuple(self._asset(row) for row in rows), revision['pack_descriptions'])

    async def _active_pack_metadata(self) -> dict | None:
        # One statement reads the head, its metadata and its pack membership
        # together, without fetching card contents or vector bytes.
        async with self.store.pool.connection() as conn:
            return await (await conn.execute('''SELECT revision.id,revision.source_root,
                revision.recipe,revision.pack_descriptions,
                ARRAY(SELECT DISTINCT alias.value->>'pack'
                    FROM sticker_catalog_items item,
                         jsonb_array_elements(item.aliases) AS alias(value)
                    WHERE item.revision_id=revision.id
                      AND NULLIF(alias.value->>'pack','') IS NOT NULL
                    ORDER BY 1) AS pack_ids
                FROM sticker_catalog_head head
                JOIN sticker_catalog_revisions revision ON revision.id=head.revision_id
                WHERE head.singleton''')).fetchone()

    async def list_pack_descriptions(self) -> dict[str, str | None]:
        """List current packs and configured IDs retained after a folder move."""
        revision = await self._active_pack_metadata()
        if revision is None:
            return {}
        descriptions = revision['pack_descriptions']
        return {pack: descriptions.get(pack) for pack in sorted(set(revision['pack_ids']) | descriptions.keys())}

    async def update_pack_descriptions(self, updates: dict[str, str | None]) -> str:
        """Update current or already configured packs; None removes a description."""
        revision = await self._active_pack_metadata()
        if revision is None:
            raise CatalogConflict('No active sticker catalog')
        unknown = updates.keys() - (set(revision['pack_ids']) | revision['pack_descriptions'].keys())
        if unknown:
            raise KeyError('Unknown sticker pack: ' + ', '.join(sorted(unknown)))
        if any(description is not None and not description.strip() for description in updates.values()):
            raise ValueError('Pack description must contain text; use None to remove it')
        descriptions = dict(revision['pack_descriptions'])
        for pack, description in updates.items():
            if description is None:
                descriptions.pop(pack, None)
            else:
                descriptions[pack] = description
        if descriptions == revision['pack_descriptions']:
            return revision['id']
        revision_id = await self.begin_revision(
            source_root=revision['source_root'], recipe=revision['recipe'],
            expected_parent=revision['id'], pack_descriptions=descriptions)
        await self.activate(revision_id)
        return revision_id

    @staticmethod
    async def _inspection_revision(conn, revision_id: str | None) -> dict | None:
        row = await (await conn.execute('''SELECT revision.*,head.revision_id AS active_revision_id
            FROM sticker_catalog_head head LEFT JOIN sticker_catalog_revisions revision
              ON revision.id=COALESCE(%s,head.revision_id) WHERE head.singleton''', (revision_id,))).fetchone()
        if not row or row['id'] is None:
            if revision_id is not None:
                raise KeyError(f'Unknown sticker revision: {revision_id}')
            return None
        return row

    async def inspect_revision(self, revision_id: str | None = None) -> dict:
        """Read publication status and saved build channels without loading vectors."""
        async with self.store.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
                revision = await self._inspection_revision(conn, revision_id)
                if revision is None:
                    return {'revision_id': None, 'active_revision_id': None, 'counts': {}}
                rows = await (await conn.execute('''SELECT state,count(*) AS assets,
                    count(*) FILTER (WHERE generated_card IS NOT NULL AND generated_card<>'null'::jsonb) AS saved_annotations,
                    count(reading_vectors_id) AS saved_reading_vectors,
                    count(image_vector_id) AS saved_image_vectors
                    FROM sticker_catalog_items WHERE revision_id=%s GROUP BY state ORDER BY state''',
                    (revision['id'],))).fetchall()
                failures = await (await conn.execute('''SELECT asset_id,aliases,error
                    FROM sticker_catalog_items WHERE revision_id=%s AND state='failed' ORDER BY asset_id''',
                    (revision['id'],))).fetchall()
        return {**{key: value for key, value in revision.items() if key != 'id'}, 'revision_id': revision['id'],
                'parent_is_current': revision['parent_id'] == revision['active_revision_id'],
                'counts': {row['state']: {key: value for key, value in row.items() if key != 'state'} for row in rows},
                'failures': failures}

    async def iter_revisions(self):
        """Stream every revision, including interrupted staging work."""
        async with self.store.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
                async with AsyncServerCursor(conn, name='sticker_revision_inspection') as cursor:
                    cursor.itersize = self.store.config.read_page_size
                    await cursor.execute('''SELECT revision.id AS revision_id,revision.parent_id,
                        revision.state,revision.source_root,revision.created_at,revision.activated_at,
                        head.revision_id AS active_revision_id,
                        revision.parent_id IS NOT DISTINCT FROM head.revision_id AS parent_is_current
                        FROM sticker_catalog_revisions revision CROSS JOIN sticker_catalog_head head
                        WHERE head.singleton ORDER BY revision.created_at DESC,revision.id DESC''')
                    async for row in cursor:
                        yield row

    async def iter_assets(self, revision_id: str | None = None, *, pack: str | None = None,
                          state: str | None = None, asset_ids: list[str] | None = None, details: bool = False):
        """Stream catalog records from one snapshot, never their binary vectors."""
        async with self.store.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
                revision = await self._inspection_revision(conn, revision_id)
                if revision is None:
                    return
                columns = ('item.asset_id,item.content_hash,item.aliases,item.media,item.generated_card,'
                    'item.corrections,item.card,item.provenance,item.dimensions,item.state,item.error,'
                    'item.image_vector_id IS NOT NULL AS has_image_vector,'
                    'item.reading_vectors_id IS NOT NULL AS has_reading_vectors'
                    if details else "item.asset_id,item.aliases,item.card->>'caption' AS caption,item.state,item.error")
                conditions = ['item.revision_id=%s']
                parameters: list[Any] = [revision['id']]
                if pack is not None:
                    conditions.append("EXISTS(SELECT 1 FROM jsonb_array_elements(item.aliases) alias WHERE alias->>'pack'=%s)")
                    parameters.append(pack)
                if state is not None:
                    conditions.append('item.state=%s')
                    parameters.append(state)
                if asset_ids is not None:
                    conditions.append('item.asset_id=ANY(%s)')
                    parameters.append(asset_ids)
                async with AsyncServerCursor(conn, name='sticker_asset_inspection') as cursor:
                    cursor.itersize = self.store.config.read_page_size
                    await cursor.execute('SELECT ' + columns + ' FROM sticker_catalog_items item WHERE '
                                         + ' AND '.join(conditions) + ' ORDER BY item.asset_id', parameters)
                    async for row in cursor:
                        packs = {alias['pack'] for alias in row['aliases']}
                        yield {'revision_id': revision['id'], **row,
                               'pack_descriptions': {pack: description for pack, description in revision['pack_descriptions'].items()
                                                     if pack in packs}}

    async def inspect_asset(self, asset_id: str, revision_id: str | None = None) -> dict:
        async with aclosing(self.iter_assets(revision_id, asset_ids=[asset_id], details=True)) as records:
            async for row in records:
                return row
        raise KeyError(f'Unknown sticker asset: {asset_id}')

    async def export_corrections(self, revision_id: str | None = None, *,
                                 asset_ids: list[str] | None = None, pack: str | None = None) -> dict[str, dict]:
        """Export complete saved overrides in the builder's existing input format."""
        result = {row['asset_id']: row['corrections'] async for row in
                  self.iter_assets(revision_id, asset_ids=asset_ids, pack=pack, details=True)}
        if asset_ids and (missing := set(asset_ids) - result.keys()):
            raise KeyError('Sticker assets not found in selection: ' + ', '.join(sorted(missing)))
        return result

    async def get_asset(self, asset_id: str, revision_id: str | None = None) -> CatalogAsset | None:
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute(ITEM_SELECT + ''' WHERE item.asset_id=%s AND
                item.revision_id=COALESCE(%s,(SELECT revision_id FROM sticker_catalog_head WHERE singleton))''',
                (asset_id, revision_id))).fetchone()
        return self._asset(row) if row else None

    async def begin_revision(self, *, source_root: str, recipe: dict, assets: list[CatalogAsset] | None = None, dimensions: int = 0, expected_parent: str | None | object = _UNSET, pack_descriptions: dict[str, str] | None = None) -> str:
        revision_id = uuid.uuid4().hex
        async with self.store.pool.connection() as conn:
            async with conn.transaction():
                head = await (await conn.execute('SELECT revision_id FROM sticker_catalog_head WHERE singleton FOR UPDATE')).fetchone()
                parent = head['revision_id']
                if expected_parent is not _UNSET and parent != expected_parent:
                    raise CatalogConflict('Active catalog changed during inventory; retry from the current catalog')
                if pack_descriptions is None:
                    parent_revision = await (await conn.execute(
                        'SELECT pack_descriptions FROM sticker_catalog_revisions WHERE id=%s',
                        (parent,))).fetchone()
                    pack_descriptions = parent_revision['pack_descriptions'] if parent_revision else {}
                await conn.execute('''INSERT INTO sticker_catalog_revisions
                    (id,parent_id,source_root,recipe,pack_descriptions) VALUES(%s,%s,%s,%s,%s)''',
                    (revision_id, parent, source_root, Jsonb(recipe), Jsonb(pack_descriptions)))
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
