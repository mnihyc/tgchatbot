"""Exercise plain-SQL deployment recovery against isolated PostgreSQL and apps."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid


REPO = Path(__file__).resolve().parents[1]
IMAGE = 'pgvector/pgvector:0.8.6-pg17-bookworm'


class DeploymentDatabaseBackupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docker = shutil.which('docker')
        if cls.docker is None:
            raise unittest.SkipTest('Docker is required for deployment database recovery tests')
        for arguments in (['compose', 'version'], ['info', '--format', '{{.ServerVersion}}'],
                          ['image', 'inspect', IMAGE]):
            if subprocess.run([cls.docker, *arguments], capture_output=True).returncode:
                raise unittest.SkipTest('Docker Compose and the PostgreSQL test image must be available')
        cls.temporary = tempfile.TemporaryDirectory(prefix='deploy-database-test-', dir=REPO / 'tests')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.project = 'tgchatbot-backup-test-' + uuid.uuid4().hex[:12]
        cls.env_bytes = b'POSTGRES_PASSWORD=synthetic-only\nDATABASE_URL=\n'
        cls.compose_bytes = f'''name: {cls.project}
services:
  postgres:
    image: {IMAGE}
    pull_policy: never
    environment:
      POSTGRES_USER: tgchatbot
      POSTGRES_DB: tgchatbot
      POSTGRES_PASSWORD: ${{POSTGRES_PASSWORD}}
    tmpfs:
      - /var/lib/postgresql/data
    healthcheck:
      test: [CMD-SHELL, "pg_isready -h 127.0.0.1 -U tgchatbot -d tgchatbot"]
      interval: 1s
      timeout: 2s
      retries: 30
  bot:
    image: {IMAGE}
    pull_policy: never
    init: true
    entrypoint: [sleep, infinity]
    environment:
      DATABASE_URL: ${{DATABASE_URL:-}}
      POSTGRES_PASSWORD: ${{POSTGRES_PASSWORD}}
  stopped-worker:
    image: {IMAGE}
    pull_policy: never
    init: true
    entrypoint: [sleep, infinity]
'''.encode()
        (cls.root / '.env').write_bytes(cls.env_bytes)
        (cls.root / 'compose.yml').write_bytes(cls.compose_bytes)
        (cls.root / 'data' / 'presets').mkdir(parents=True)
        (cls.root / 'data' / 'presets' / 'role.txt').write_bytes(b'Operator-owned role preset.\n')
        (cls.root / 'data' / 'attachment.bin').write_bytes(b'Operator-owned attachment\x00\xff')
        cls.addClassCleanup(cls._stop_fixture)
        cls.compose('up', '-d', '--wait', '--wait-timeout', '40', 'postgres', 'bot')
        cls.compose('create', 'stopped-worker')
        # Forward Docker while retaining the older Compose v2 start interface.
        # Record updater commands to prove no download/build/activation occurs.
        cls.bin = cls.root / 'bin'
        cls.bin.mkdir()
        wrapper = cls.bin / 'docker'
        wrapper.write_text('#!' + shutil.which('python3') + '\n' + f'''
import json, os, subprocess, sys
from pathlib import Path
arguments = sys.argv[1:]
with Path(os.environ['DEPLOY_TEST_CALLS']).open('a') as stream:
    stream.write(json.dumps(arguments) + '\\n')
if arguments[:2] == ['compose', 'start'] and '--wait' in arguments:
    print('unknown flag: --wait', file=sys.stderr)
    raise SystemExit(125)
if os.environ.get('DEPLOY_TEST_FAIL_DUMP') == '1' and any('pg_dump' in item for item in arguments):
    print('-- incomplete synthetic dump')
    raise SystemExit(23)
if os.environ.get('DEPLOY_TEST_FAIL_STOP') == '1' and arguments[:2] == ['compose', 'stop']:
    subprocess.run([{cls.docker!r}, *arguments], check=True)
    raise SystemExit(23)
os.execv({cls.docker!r}, [{cls.docker!r}, *arguments])
''')
        wrapper.chmod(0o755)
        curl = cls.bin / 'curl'
        curl.write_text('#!/bin/sh\necho "Database command unexpectedly requested a download" >&2\nexit 97\n')
        curl.chmod(0o755)
        reader = cls.bin / 'cat'
        reader.write_text('#!' + shutil.which('python3') + '\n' + f'''
import os, sys
from pathlib import Path
arguments = sys.argv[1:]
if os.environ.get('DEPLOY_TEST_FAIL_READ') == '1' and arguments:
    sys.stdout.buffer.write(Path(arguments[-1]).read_bytes())
    raise SystemExit(24)
os.execv({shutil.which('cat')!r}, [{shutil.which('cat')!r}, *arguments])
''')
        reader.chmod(0o755)

    @classmethod
    def compose(cls, *arguments, input=None, check=True):
        result = subprocess.run([cls.docker, 'compose', *arguments], cwd=cls.root,
            input=input, text=True, capture_output=True, timeout=60)
        if check and result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result

    @classmethod
    def _stop_fixture(cls):
        cls.compose('down', '--volumes', '--remove-orphans', check=False)

    @classmethod
    def sql(cls, statement):
        return cls.compose('exec', '-T', 'postgres', 'psql', '-X', '-qAt',
            '-v', 'ON_ERROR_STOP=1', '-U', 'tgchatbot', '-d', 'tgchatbot', '-f', '-', input=statement).stdout

    def setUp(self):
        shutil.copy2(REPO / 'deploy' / 'update.sh', self.root / 'update.sh')
        self.calls_path = self.root / ('calls-' + uuid.uuid4().hex + '.jsonl')
        self.environment = dict(os.environ, PATH=f'{self.bin}:{os.environ.get("PATH", "")}',
            DEPLOY_TEST_CALLS=str(self.calls_path))
        self.compose('start', 'postgres', 'bot')
        for _ in range(100):
            ready = self.compose('exec', '-T', 'postgres', 'pg_isready', '-h', '127.0.0.1',
                '-U', 'tgchatbot', '-d', 'tgchatbot', check=False)
            if ready.returncode == 0:
                break
            time.sleep(0.1)
        else:
            self.fail('PostgreSQL did not restart')
        self.compose('stop', 'stopped-worker')
        self.sql('''DROP SCHEMA IF EXISTS public CASCADE;
            DROP SCHEMA IF EXISTS custom CASCADE;
            DROP SCHEMA IF EXISTS post_backup CASCADE;
            CREATE SCHEMA public AUTHORIZATION pg_database_owner;
            CREATE EXTENSION vector;
            CREATE TABLE public.messages(id int PRIMARY KEY, body text, embedding vector(3));
            INSERT INTO public.messages VALUES(1, '你好，保留原始消息。', '[1,2,3]');
            CREATE TABLE public.profiles(actor_id text PRIMARY KEY, facts jsonb);
            INSERT INTO public.profiles VALUES('person_id:101', '{"preference":"简短自然"}');
            CREATE SCHEMA custom;
            CREATE TABLE custom.note(body text);
            INSERT INTO custom.note VALUES('保留自定义数据');''')
        self.files = {str(path.relative_to(self.root)): path.read_bytes() for path in
            (self.root / '.env', self.root / 'compose.yml',
             self.root / 'data' / 'presets' / 'role.txt', self.root / 'data' / 'attachment.bin')}

    def update(self, *arguments, success=True, **extra):
        result = subprocess.run(['bash', str(self.root / 'update.sh'), *arguments],
            cwd=self.root, env=dict(self.environment, **extra), text=True,
            capture_output=True, timeout=90)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def state(self):
        return json.loads(self.sql('''SELECT json_build_object(
            'messages',(SELECT json_agg(messages ORDER BY id) FROM public.messages),
            'profiles',(SELECT json_agg(profiles ORDER BY actor_id) FROM public.profiles),
            'custom',(SELECT json_agg(note ORDER BY body) FROM custom.note),
            'tables',(SELECT json_agg(schemaname||'.'||tablename ORDER BY schemaname,tablename)
                FROM pg_tables WHERE schemaname !~ '^pg_' AND schemaname <> 'information_schema'),
            'schemas',(SELECT json_agg(nspname ORDER BY nspname) FROM pg_namespace
                WHERE nspname !~ '^pg_' AND nspname <> 'information_schema'));'''))

    def running(self):
        return set(self.compose('ps', '--services', '--status', 'running').stdout.split())

    def assert_boundary(self):
        for relative, original in self.files.items():
            self.assertEqual((self.root / relative).read_bytes(), original, relative)
        self.assertEqual(self.running(), {'postgres', 'bot'},
            'Restore must resume the previously running bot without starting the stopped worker')
        calls = [json.loads(line) for line in self.calls_path.read_text().splitlines()]
        self.assertFalse(any(call and call[0] in {'build', 'pull', 'tag', 'load', 'run', 'image'}
                             for call in calls), calls)

    def test_default_plain_sql_backup_and_restore_replace_original_state_without_touching_files(self):
        before = self.state()
        self.update('backup')
        candidates = list((self.root / 'backups').glob('database-*.sql'))
        self.assertEqual(len(candidates), 1)
        backup = candidates[0]
        self.assertRegex(backup.name, r'^database-\d{8}T\d{6}Z(?:-\d+)?\.sql$')
        self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
        text = backup.read_text()
        self.assertIn('PostgreSQL database dump', text)
        self.assertIn('你好，保留原始消息。', text)
        self.assertIn('简短自然', text)
        self.sql('''UPDATE public.messages SET body='changed', embedding='[3,2,1]';
            UPDATE public.profiles SET facts='{"wrong":true}';
            UPDATE custom.note SET body='changed';
            CREATE TABLE public.later(id int); INSERT INTO public.later VALUES(99);
            CREATE SCHEMA post_backup; CREATE TABLE post_backup.later(id int);''')
        self.update('restore', str(backup))
        self.assertEqual(self.state(), before)
        self.assert_boundary()

    def test_late_sql_error_rolls_back_and_restores_running_services(self):
        backup = self.root / 'recovery with spaces.sql'
        self.update('backup', str(backup))
        self.sql("UPDATE public.messages SET body='current message must survive'; "
                 "CREATE SCHEMA post_backup; CREATE TABLE post_backup.later(id int);")
        before = self.state()
        broken = self.root / 'invalid after valid rows.sql'
        broken.write_bytes(backup.read_bytes() + b'\nSELECT * FROM deliberately_missing_restore_table;\n')
        self.update('restore', str(broken), success=False)
        self.assertEqual(self.state(), before)
        # A producer can fail even after providing valid SQL. Premature EOF
        # must not commit the already executed reset/dump statements.
        self.update('restore', str(backup), success=False, DEPLOY_TEST_FAIL_READ='1')
        self.assertEqual(self.state(), before)
        self.assert_boundary()

    def test_failed_dump_preserves_existing_backup_and_operation_lock_prevents_restore(self):
        backup = self.root / 'preserved.sql'
        backup.write_bytes(b'Existing recovery copy must survive.\n')
        self.update('backup', str(backup), success=False, DEPLOY_TEST_FAIL_DUMP='1')
        self.assertEqual(backup.read_bytes(), b'Existing recovery copy must survive.\n')
        self.assertEqual(list(self.root.glob('preserved.sql.partial.*')), [])
        before = self.state()
        prior = len(self.calls_path.read_text().splitlines())
        stopped = self.update('restore', str(backup), success=False, DEPLOY_TEST_FAIL_STOP='1')
        self.assertIn('restore has not started', stopped.stderr)
        calls = [json.loads(line) for line in self.calls_path.read_text().splitlines()[prior:]]
        self.assertFalse(any('psql' in part for call in calls for part in call))
        self.assertEqual(self.state(), before)
        self.assert_boundary()
        (self.root / 'tmp').mkdir(exist_ok=True)
        with (self.root / 'tmp' / 'update.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.update('restore', str(backup), success=False)
        self.assertEqual(self.state(), before)
        self.assert_boundary()

    def test_database_must_already_run_and_stopped_app_remains_stopped(self):
        self.compose('stop', 'postgres', 'bot')
        self.addCleanup(self.compose, 'start', 'postgres', 'bot')
        target = self.root / 'unavailable.sql'
        self.update('backup', str(target), success=False)
        self.assertFalse(target.exists())
        self.assertEqual(self.running(), set())
        self.compose('start', 'postgres')
        for _ in range(100):
            if self.compose('exec', '-T', 'postgres', 'pg_isready', '-h', '127.0.0.1',
                    '-U', 'tgchatbot', '-d', 'tgchatbot', check=False).returncode == 0:
                break
            time.sleep(0.1)
        self.update('backup', str(target))
        self.update('restore', str(target))
        self.assertEqual(self.running(), {'postgres'})


if __name__ == '__main__':
    unittest.main()
