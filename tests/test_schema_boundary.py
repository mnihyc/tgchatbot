"""A new development schema must never silently rewrite an older deployment."""
from tests.business_helpers import BusinessTestCase
from tgchatbot.storage.postgres_store import PostgresStore


class SchemaBoundaryTests(BusinessTestCase):
    async def test_older_schema_is_rejected_without_rewriting_existing_records(self):
        await self.settings()
        async with self.store.pool.connection() as conn:
            await conn.execute('UPDATE schema_version SET version=2')
        other = PostgresStore(self.test_dsn, schema=self.schema)
        try:
            with self.assertRaisesRegex(RuntimeError, 'incompatible conversation schema'):
                await other.initialize()
            async with self.store.pool.connection() as conn:
                version = await (await conn.execute('SELECT version FROM schema_version')).fetchone()
                session = await (await conn.execute('SELECT session_id FROM sessions')).fetchone()
            self.assertEqual(version['version'], 2)
            self.assertEqual(session['session_id'], self.session)
        finally:
            await other.close()
