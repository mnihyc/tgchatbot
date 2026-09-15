"""Exercise the real simple updater with disposable releases and mocked Docker/network."""
from __future__ import annotations

import fcntl
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
MOCK = r'''#!/usr/bin/env python3
import hashlib, json, os, pathlib, re, runpy, sys
args = sys.argv[1:]
name = pathlib.Path(sys.argv[0]).name
with open(os.environ["MOCK_LOG"], "a") as out:
    out.write(json.dumps([name, *args]) + "\n")
if name == "curl":
    if "--write-out" in args:
        print("https://github.com/example/chatbot/releases/tag/v0.2.0", end="")
    else:
        tag, filename = args[-1].split("/")[-2:]
        output = pathlib.Path(args[args.index("--output") + 1])
        output.write_bytes((pathlib.Path(os.environ["MOCK_ASSETS"]) / tag / filename).read_bytes())
    sys.exit(0)
state_path = pathlib.Path(os.environ["MOCK_STATE"])
state = json.loads(state_path.read_text()) if state_path.exists() else {"tags": {}, "images": {}}
def image_id(reference):
    return state["tags"].get(reference, reference if reference in state["images"] else None)
if args[:1] == ["compose"] and "config" in args and "json" in args:
    # The real parser is Compose. Supply its shape here so the actual released
    # credential helper, rather than a mocked initializer, handles the .env file.
    values = {}
    for line in pathlib.Path('.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip("\"'")
    print(json.dumps({'services': {'bot': {'environment': values}}}))
    sys.exit(0)
elif args[:1] == ["run"] and args[-1] == '/usr/local/lib/tgchatbot-deploy-configure.py':
    mounts = [args[index + 1] for index, value in enumerate(args) if value == '-v']
    deployment_path = pathlib.Path(next(value.split(':', 1)[0] for value in mounts if value.endswith(':/deployment')))
    env_path = deployment_path / '.env'
    data_path = pathlib.Path(next(value.split(':', 1)[0] for value in mounts if ':/deployment/data' in value))
    helper = runpy.run_path(os.environ['DEPLOY_HELPER_SOURCE'])
    print(helper['configure'](json.load(sys.stdin), env_path, data_path))
    sys.exit(0)
elif args[:1] == ["compose"] and '--check-schema' in args:
    identity = image_id(os.environ['TGCHATBOT_IMAGE'])
    tag = state['images'][identity]['tag']
    if tag in (os.environ.get('FAIL_SCHEMA_TAG'), state.get('incompatible_schema_tag')):
        sys.exit(1)
elif args[:1] == ['compose'] and 'config' in args and '--services' in args:
    path = pathlib.Path(args[args.index('-f') + 1]) if '-f' in args else pathlib.Path('compose.yml')
    section = path.read_text().split('\nservices:\n', 1)[1]
    print('\n'.join(name for name in re.findall(r'^  ([a-zA-Z0-9_-]+):', section, re.M) if name != 'postgres'))
    sys.exit(0)
elif args[:2] == ["image", "inspect"]:
    identity = image_id(args[-1])
    if identity is None:
        sys.exit(1)
    if "--format" in args:
        field = args[args.index("--format") + 1]
        if field == "{{.Id}}":
            print(identity)
        elif "image.version" in field:
            print(state["images"][identity]["tag"])
        else:
            print(os.environ.get("MOCK_COMMIT", state["images"][identity]["commit"]))
elif args and args[0] == "build":
    context = pathlib.Path(args[-1])
    build_args = dict(args[index + 1].split('=', 1) for index, value in enumerate(args) if value == '--build-arg')
    tag = build_args['RELEASE_TAG']
    if tag == os.environ.get('FAIL_BUILD_TAG'):
        sys.exit(1)
    files = {str(path.relative_to(context)): path.read_text() for path in context.rglob('*') if path.is_file()}
    payload = {'tag': tag, 'commit': build_args['RELEASE_COMMIT'], 'build_files': files}
    identity = "sha256:" + hashlib.sha256(tag.encode()).hexdigest()
    state['images'][identity] = payload
    state['tags'][args[args.index('--tag') + 1]] = identity
elif args and args[0] == "tag":
    identity = image_id(args[1])
    if identity is None:
        sys.exit(1)
    state["tags"][args[2]] = identity
elif args and args[0] == "compose" and "up" in args:
    if args[-1] != 'postgres':
        identity = image_id("tgchatbot:current")
        if identity is None or state["images"][identity]["tag"] == os.environ.get("FAIL_TAG"):
            state['incompatible_schema_tag'] = os.environ.get('INCOMPATIBLE_AFTER_FAILURE')
            state_path.write_text(json.dumps(state))
            sys.exit(1)
state_path.write_text(json.dumps(state))
sys.exit(0)
'''


class ReleaseUpdaterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="deploy-test-", dir=REPO / "tests")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.install = self.path / "install"
        self.install.mkdir()
        (self.install / "data" / "tantivy_index").mkdir(parents=True)
        self.state = self.install / "data" / "tgchatbot.sqlite3"
        self.state.write_bytes(b"existing business state")
        (self.install / "data" / "tantivy_index" / "meta.json").write_text('{"preserve": true}')
        self.env_content = "TGBOT_TOKEN=secret\nDEFAULT_SYSTEM_PROMPT='keep $cash and quotes'\nUNUSED=$(touch SHOULD_NOT_EXIST)\n"
        (self.install / ".env").write_text(self.env_content)
        shutil.copy2(REPO / "deploy" / "update.sh", self.install / "update.sh")
        self.bin = self.path / "bin"
        self.bin.mkdir()
        for name in ("docker", "curl"):
            executable = self.bin / name
            executable.write_text(MOCK)
            executable.chmod(0o755)
        self.assets = self.path / "assets"
        self.log = self.path / "calls.jsonl"
        self.docker_state = self.path / "docker.json"
        self.environment = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                                MOCK_LOG=str(self.log), MOCK_ASSETS=str(self.assets),
                                MOCK_STATE=str(self.docker_state), TGCHATBOT_RELEASE_REPO="example/chatbot",
                                DEPLOY_HELPER_SOURCE=str(REPO / 'deploy' / 'configure_database.py'))
        for tag in ("v0.1.0", "v0.2.0"):
            self.make_release(tag)

    def make_release(self, tag, *, compose=None):
        directory = self.assets / tag
        directory.mkdir(parents=True, exist_ok=True)
        with tarfile.open(directory / f"tgchatbot-deploy-{tag}.tar.gz", "w:gz") as archive:
            files = {"RELEASE_TAG": tag.encode(), "RELEASE_COMMIT": COMMIT.encode(),
                     ".env.example": b"TGBOT_TOKEN=\nOPENAI_API_KEY=\n"}
            for name in ("compose.yml", "update.sh"):
                files[name] = (REPO / "deploy" / name).read_bytes()
            if compose is not None:
                files['compose.yml'] = compose
            for name in ("Dockerfile", ".dockerignore", "deploy/entrypoint.sh", "deploy/configure_database.py"):
                files[name] = (REPO / name).read_bytes()
            files["build/runtime-requirements.txt"] = b"locked third-party dependencies fixture"
            files[f"build/tgchatbot-{tag[1:]}-py3-none-any.whl"] = b"portable application wheel fixture"
            for name, contents in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))

    def run_update(self, target=None, *, success=True, **env):
        command = ["bash", str(self.install / "update.sh")]
        if target is not None:
            command.append(target)
        result = subprocess.run(command, cwd=self.install, env=dict(self.environment, **env),
                                text=True, capture_output=True, timeout=15)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def image_tag(self, alias):
        state = json.loads(self.docker_state.read_text())
        return state["images"][state["tags"][alias]]["tag"]

    def operator_compose(self):
        content = (REPO / 'deploy/compose.yml').read_text().replace(
            '  postgres:\n', '  postgres:\n    ports: ["127.0.0.1:15432:5432"]\n', 1)
        content += '\n  operator-worker:\n    image: tgchatbot:current\n    mem_limit: 256m\n'
        content += '\n# Operator-maintained networking and services.\n'
        encoded = content.encode()
        (self.install / 'compose.yml').write_bytes(encoded)
        return encoded

    def assert_compose_preserved(self, expected):
        self.assertEqual((self.install / 'compose.yml').read_bytes(), expected)

    def assert_only_local_compose_used(self, calls):
        for call in calls:
            if call[:2] == ['docker', 'compose'] and '-f' in call:
                self.assertEqual((self.install / call[call.index('-f') + 1]).resolve(),
                                 (self.install / 'compose.yml').resolve())

    def assert_simple_layout(self):
        self.assertFalse((self.install / "runtime").exists())
        self.assertFalse((self.install / "RELEASE_TAG").exists())
        self.assertFalse(list(self.install.rglob("release.env")))
        scratch = self.install / "tmp" / "update"
        self.assertEqual(list(scratch.iterdir()), [])

    def assert_data_preserved(self):
        self.assertEqual(self.state.read_bytes(), b"existing business state")
        current = (self.install / ".env").read_text()
        self.assertTrue(current.startswith(self.env_content))
        if current != self.env_content:
            self.assertRegex(current[len(self.env_content):], r'^POSTGRES_PASSWORD=[0-9a-f]{64}\n$')
        self.assertEqual((self.install / "data" / "tantivy_index" / "meta.json").read_text(), '{"preserve": true}')
        self.assertFalse((self.install / "SHOULD_NOT_EXIST").exists())

    def test_no_argument_resolves_latest_then_uses_plain_compose_and_cleans_scratch(self):
        result = self.run_update()
        self.assertNotIn("Fill in", result.stdout)
        self.assertNotIn("Created .env", result.stdout)
        self.assertEqual(self.image_tag("tgchatbot:current"), "v0.2.0")
        urls = [call[-1] for call in self.calls() if call[0] == "curl"]
        self.assertTrue(urls[0].endswith("/releases/latest"))
        self.assertTrue(all("/releases/download/v0.2.0/" in url for url in urls[1:]))
        up = next(call for call in self.calls() if "up" in call and "bot" in call)
        self.assertIn("--no-build", up)
        self.assertIn("--wait", up)
        self.assertIn("never", up)
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_bootstrap_creates_configuration_without_building_or_starting_application(self):
        (self.install / ".env").unlink()
        (self.install / "update.sh").chmod(0o644)
        result = self.run_update()
        self.assertEqual((self.install / ".env").read_text(), "TGBOT_TOKEN=\nOPENAI_API_KEY=\n")
        self.assertEqual((self.install / "compose.yml").read_bytes(),
                         (REPO / 'deploy/compose.yml').read_bytes())
        self.assertTrue(os.access(self.install / "update.sh", os.X_OK))
        self.assertFalse(any("build" in call or "up" in call for call in self.calls()))
        self.assertFalse(any("tgchatbot-linux-" in call[-1] for call in self.calls() if call[0] == "curl"))
        self.assert_simple_layout()

    def test_bootstrap_preserves_existing_operator_compose_when_env_is_missing(self):
        expected = self.operator_compose()
        (self.install / '.env').unlink()
        self.run_update()
        self.assert_compose_preserved(expected)
        self.assertFalse(any('build' in call or 'up' in call for call in self.calls()))
        self.assert_simple_layout()

    def test_bundled_release_updater_takes_over_before_obsolete_activation_logic(self):
        installed = self.install / 'update.sh'
        installed.write_text(installed.read_text().replace(
            'start() {', "start() { fail 'obsolete activation must never execute';", 1))
        result = self.run_update()
        self.assertIn('Continuing with the release updater', result.stdout)
        self.assertEqual(installed.read_bytes(), (REPO / 'deploy' / 'update.sh').read_bytes())
        self.assertEqual(self.image_tag('tgchatbot:current'), 'v0.2.0')
        self.assertEqual(len([call for call in self.calls() if call[:2] == ['docker', 'build']]), 1)
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_update_uses_operator_services_and_preserves_networking_despite_new_release_template(self):
        self.run_update('v0.1.0')
        expected = self.operator_compose()
        released = (REPO / 'deploy/compose.yml').read_bytes()
        released += b'\n  new-release-service:\n    image: tgchatbot:current\n'
        self.make_release('v0.2.0', compose=released)
        prior = len(self.calls())
        self.run_update('v0.2.0')
        calls = self.calls()[prior:]
        stopped = next(i for i, call in enumerate(calls) if 'stop' in call and 'operator-worker' in call)
        started = next(i for i, call in enumerate(calls) if 'up' in call and 'bot' in call)
        self.assertLess(stopped, started)
        self.assertIn('operator-worker', calls[started])
        self.assertFalse(any('new-release-service' in call for call in calls))
        self.assertFalse(any('rm' in call and 'operator-worker' in call for call in calls))
        self.assert_compose_preserved(expected)
        self.assertFalse((self.install / 'compose.previous.yml').exists())
        self.assert_only_local_compose_used(calls)
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_failed_replacement_recovers_prior_image_using_unchanged_operator_services(self):
        self.run_update('v0.1.0')
        expected = self.operator_compose()
        prior = len(self.calls())
        self.run_update('v0.2.0', success=False, FAIL_TAG='v0.2.0')
        calls = self.calls()[prior:]
        self.assertFalse(any('rm' in call and 'operator-worker' in call for call in calls))
        self.assert_compose_preserved(expected)
        app_starts = [call for call in calls if 'up' in call and 'bot' in call]
        self.assertEqual(len(app_starts), 2)
        self.assertTrue(all('operator-worker' in call for call in app_starts))
        self.assertEqual(self.image_tag('tgchatbot:current'), 'v0.1.0')
        self.assert_only_local_compose_used(calls)
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_unreadable_bundle_never_builds_or_stops_services(self):
        archive = self.assets / "v0.2.0" / "tgchatbot-deploy-v0.2.0.tar.gz"
        archive.write_bytes(b"corrupted code bundle")
        self.run_update("v0.2.0", success=False)
        self.assertFalse(any("build" in call or "stop" in call or "up" in call for call in self.calls()))
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_docker_build_receives_only_released_code_and_locked_dependency_inputs(self):
        stale = self.install / 'tmp' / 'update' / 'build-context' / 'build'
        stale.mkdir(parents=True)
        (stale / 'tgchatbot-0.0.0-py3-none-any.whl').write_bytes(b'interrupted prior update')
        self.run_update('v0.2.0')
        state = json.loads(self.docker_state.read_text())
        image = state['images'][state['tags']['tgchatbot:current']]
        files = image['build_files']
        self.assertEqual(set(files), {
            'Dockerfile', '.dockerignore', 'build/runtime-requirements.txt',
            'build/tgchatbot-0.2.0-py3-none-any.whl',
            'deploy/entrypoint.sh', 'deploy/configure_database.py',
        })
        self.assertEqual(files['build/tgchatbot-0.2.0-py3-none-any.whl'], 'portable application wheel fixture')
        self.assertEqual(files['build/runtime-requirements.txt'], 'locked third-party dependencies fixture')
        self.assertFalse(any(call[:2] == ['docker', 'load'] for call in self.calls()))
        downloads = [call[-1].split('/')[-1] for call in self.calls() if call[0] == 'curl']
        self.assertEqual(set(downloads), {'tgchatbot-deploy-v0.2.0.tar.gz'})
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_failed_dependency_install_keeps_working_image_and_services_untouched(self):
        self.run_update('v0.1.0')
        old_compose = self.operator_compose()
        before = len(self.calls())
        self.run_update('v0.2.0', success=False, FAIL_BUILD_TAG='v0.2.0')
        self.assertEqual(self.image_tag('tgchatbot:current'), 'v0.1.0')
        self.assertEqual((self.install / 'compose.yml').read_bytes(), old_compose)
        self.assertFalse(any('stop' in call or 'up' in call for call in self.calls()[before:]))
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_wrong_image_commit_never_stops_services(self):
        self.run_update("v0.2.0", success=False, MOCK_COMMIT="b" * 40)
        self.assertFalse(any("stop" in call or "up" in call for call in self.calls()))
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_manual_rollback_swaps_images_without_network_or_deployment_metadata(self):
        self.run_update("v0.1.0")
        self.run_update("v0.2.0")
        expected = self.operator_compose()
        self.assertFalse((self.install / 'compose.previous.yml').exists())
        downloads_before = len([call for call in self.calls() if call[0] == "curl"])
        self.run_update("rollback")
        self.assertEqual(self.image_tag("tgchatbot:current"), "v0.1.0")
        self.assertEqual(self.image_tag("tgchatbot:previous"), "v0.2.0")
        self.assertEqual(len([call for call in self.calls() if call[0] == "curl"]), downloads_before)
        self.assert_compose_preserved(expected)
        self.assertIn('operator-worker', [call for call in self.calls() if 'up' in call and 'bot' in call][-1])
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_stale_compose_history_never_overrides_operator_configuration_during_rollback(self):
        self.run_update('v0.1.0')
        self.run_update('v0.2.0')
        expected = self.operator_compose()
        stale = self.install / 'compose.previous.yml'
        old = b'# Retained artifact from a former updater; not active configuration.\n'
        stale.write_bytes(old)
        self.run_update('rollback')
        self.assert_compose_preserved(expected)
        self.assertEqual(stale.read_bytes(), old)
        self.assertEqual(self.image_tag('tgchatbot:current'), 'v0.1.0')
        self.assert_only_local_compose_used(self.calls())
        self.assert_data_preserved()

    def test_failed_first_start_stops_services_and_retains_data(self):
        self.run_update("v0.1.0", success=False, FAIL_TAG="v0.1.0")
        self.assertTrue(any("stop" in call for call in self.calls()))
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_same_release_update_keeps_previous_version_available(self):
        self.run_update("v0.1.0")
        self.run_update("v0.2.0")
        self.run_update("v0.2.0")
        self.assertEqual(self.image_tag("tgchatbot:previous"), "v0.1.0")
        self.assert_simple_layout()

    def test_local_database_password_is_created_once_and_database_precedes_bot(self):
        self.run_update('v0.1.0')
        first_env = (self.install / '.env').read_text()
        self.run_update('v0.2.0')
        self.assertEqual((self.install / '.env').read_text(), first_env)
        calls = self.calls()
        database_start = next(i for i, call in enumerate(calls) if 'up' in call and call[-1] == 'postgres')
        app_start = next(i for i, call in enumerate(calls) if 'up' in call and 'bot' in call)
        self.assertLess(database_start, app_start)
        self.assertTrue(any('pull' in call and call[-1] == 'postgres' for call in calls))
        helper = next(call for call in calls if call[-1] == '/usr/local/lib/tgchatbot-deploy-configure.py')
        self.assertIn(f'{self.install}:/deployment', helper)
        self.assertIn(f'{self.install}/data:/deployment/data:ro', helper)
        self.assert_data_preserved()

    def test_failed_credential_commit_preserves_original_keys_and_removes_temporary_copy(self):
        env_path = self.install / '.env'
        original = env_path.read_bytes()
        env_path.chmod(0o640)
        before = env_path.stat()
        configure = runpy.run_path(REPO / 'deploy' / 'configure_database.py')['configure']
        configuration = {'services': {'bot': {'environment': {'TGBOT_TOKEN': 'secret'}}}}

        def failed_rename(temporary, destination):
            # The complete replacement has been written before the failure;
            # existing API keys and file ownership must still be untouched.
            self.assertEqual(destination, env_path)
            self.assertTrue(Path(temporary).read_bytes().startswith(original))
            self.assertIn(b'POSTGRES_PASSWORD=', Path(temporary).read_bytes())
            self.assertEqual(env_path.read_bytes(), original)
            raise OSError('simulated filesystem failure immediately before rename')

        with patch('os.replace', side_effect=failed_rename):
            with self.assertRaisesRegex(OSError, 'immediately before rename'):
                configure(configuration, env_path, self.install / 'data')
        self.assertEqual(env_path.read_bytes(), original)
        self.assertEqual((env_path.stat().st_uid, env_path.stat().st_gid, env_path.stat().st_mode),
                         (before.st_uid, before.st_gid, before.st_mode))
        self.assertEqual(list(self.install.glob('.env-update-*')), [])

        self.assertEqual(configure(configuration, env_path, self.install / 'data'), 'local')
        self.assertTrue(env_path.read_bytes().startswith(original))
        self.assertEqual((env_path.stat().st_uid, env_path.stat().st_gid, env_path.stat().st_mode),
                         (before.st_uid, before.st_gid, before.st_mode))
        self.assertEqual(list(self.install.glob('.env-update-*')), [])

    def test_real_compose_uses_created_credential_for_bot_and_database(self):
        docker = shutil.which('docker')
        if not docker or subprocess.run([docker, 'compose', 'version'], capture_output=True).returncode:
            self.skipTest('Docker Compose is unavailable')
        shutil.copy2(REPO / 'deploy' / 'compose.yml', self.install / 'compose.yml')
        original = b"TGBOT_TOKEN=fixture\r\nDEFAULT_SYSTEM_PROMPT='literal $cash $(not-a-command)\r\nPOSTGRES_PASSWORD=prompt-text'\r\n"
        (self.install / '.env').write_bytes(original)
        command = [docker, 'compose', '--project-directory', str(self.install), '-f', str(self.install / 'compose.yml')]

        def config(*extra):
            return json.loads(subprocess.check_output([*command, *extra, 'config', '--format', 'json'], stderr=subprocess.PIPE))

        before = config()
        configure = runpy.run_path(REPO / 'deploy' / 'configure_database.py')['configure']
        self.assertEqual(configure(before, self.install / '.env', self.install / 'data'), 'local')
        self.assertTrue((self.install / '.env').read_bytes().startswith(original))
        default = config()
        local = config('--profile', 'local-database')
        self.assertNotIn('postgres', default['services'])
        bot_env = local['services']['bot']['environment']
        self.assertEqual(bot_env['POSTGRES_PASSWORD'], local['services']['postgres']['environment']['POSTGRES_PASSWORD'])
        self.assertRegex(bot_env['POSTGRES_PASSWORD'], r'^[0-9a-f]{64}$')
        self.assertEqual(bot_env['DEFAULT_SYSTEM_PROMPT'].replace('$$', '$'), 'literal $cash $(not-a-command)\r\nPOSTGRES_PASSWORD=prompt-text')

    def test_real_compose_renders_operator_postgres_port_after_application_update(self):
        docker = shutil.which('docker')
        if not docker or subprocess.run([docker, 'compose', 'version'], capture_output=True).returncode:
            self.skipTest('Docker Compose is unavailable')
        expected = self.operator_compose()
        self.run_update('v0.2.0')
        configuration = json.loads(subprocess.check_output([
            docker, 'compose', '--project-directory', str(self.install),
            '--profile', 'local-database', 'config', '--format', 'json'], stderr=subprocess.PIPE))
        port = configuration['services']['postgres']['ports'][0]
        self.assertEqual((port['host_ip'], port['published'], port['target']),
                         ('127.0.0.1', '15432', 5432))
        self.assertIn('operator-worker', configuration['services'])
        self.assert_compose_preserved(expected)

    def test_external_database_does_not_start_local_postgres_or_create_password(self):
        self.env_content += 'DATABASE_URL=postgresql://fixture:secret@database.example/tgchatbot\n'
        (self.install / '.env').write_text(self.env_content)
        self.run_update()
        self.assertEqual((self.install / '.env').read_text(), self.env_content)
        self.assertFalse(any(('up' in call or 'pull' in call) and call[-1] == 'postgres' for call in self.calls()))
        self.assert_data_preserved()

    def test_existing_database_without_password_never_gets_a_replacement(self):
        expected = self.operator_compose()
        cluster = self.install / 'data' / 'postgres'
        cluster.mkdir()
        (cluster / 'PG_VERSION').write_text('17\n')
        result = self.run_update(success=False)
        self.assertIn('original POSTGRES_PASSWORD', result.stderr)
        self.assertEqual((self.install / '.env').read_text(), self.env_content)
        self.assertFalse(any('up' in call or 'stop' in call for call in self.calls()))
        self.assert_compose_preserved(expected)

    def test_failed_external_database_preflight_does_not_stop_previous_local_database(self):
        self.run_update('v0.1.0')
        previous_calls = len(self.calls())
        with (self.install / '.env').open('a') as stream:
            stream.write('DATABASE_URL=postgresql://fixture:secret@database.example/tgchatbot\n')
        self.environment['FAIL_SCHEMA_TAG'] = 'v0.2.0'
        self.run_update('v0.2.0', success=False)
        self.assertFalse(any('stop' in call for call in self.calls()[previous_calls:]))
        self.assertEqual(self.image_tag('tgchatbot:current'), 'v0.1.0')

    def test_incompatible_candidate_does_not_stop_working_application(self):
        self.run_update('v0.1.0')
        expected = self.operator_compose()
        prior_calls = len(self.calls())
        self.run_update('v0.2.0', success=False, FAIL_SCHEMA_TAG='v0.2.0')
        self.assertEqual(self.image_tag('tgchatbot:current'), 'v0.1.0')
        self.assertFalse(any('stop' in call and 'bot' in call for call in self.calls()[prior_calls:]))
        self.assert_compose_preserved(expected)

    def test_failed_activation_cannot_restart_prior_image_after_incompatible_schema_change(self):
        self.run_update('v0.1.0')
        expected = self.operator_compose()
        prior_calls = len(self.calls())
        result = self.run_update('v0.2.0', success=False, FAIL_TAG='v0.2.0', INCOMPATIBLE_AFTER_FAILURE='v0.1.0')
        self.assertIn('no compatible prior image', result.stderr)
        app_starts = [call for call in self.calls()[prior_calls:] if 'up' in call and 'bot' in call]
        self.assertEqual(len(app_starts), 1)
        self.assertEqual(self.image_tag('tgchatbot:current'), 'v0.2.0')
        self.assert_compose_preserved(expected)
        self.assert_data_preserved()

    def test_manual_rollback_checks_schema_before_stopping_current_application(self):
        self.run_update('v0.1.0')
        self.run_update('v0.2.0')
        expected = self.operator_compose()
        prior_calls = len(self.calls())
        self.run_update('rollback', success=False, FAIL_SCHEMA_TAG='v0.1.0')
        self.assertEqual(self.image_tag('tgchatbot:current'), 'v0.2.0')
        self.assertFalse(any('stop' in call and 'bot' in call for call in self.calls()[prior_calls:]))
        self.assert_compose_preserved(expected)

    def test_previous_image_is_saved_before_current_changes(self):
        self.run_update("v0.1.0")
        prior_calls = len(self.calls())
        self.run_update("v0.2.0")
        calls = self.calls()[prior_calls:]
        save_previous = next(i for i, call in enumerate(calls) if call[:2] == ["docker", "tag"] and call[-1] == "tgchatbot:previous")
        replace_current = next(i for i, call in enumerate(calls) if call[:2] == ["docker", "tag"] and call[-1] == "tgchatbot:current")
        self.assertLess(save_previous, replace_current)

    def test_malformed_release_target_has_no_network_or_docker_side_effects(self):
        self.run_update("../../other", success=False)
        self.assertEqual(self.calls(), [])

    def test_concurrent_update_does_not_remove_first_updaters_download(self):
        scratch = self.install / "tmp" / "update"
        scratch.mkdir(parents=True)
        downloading = scratch / "deploy.tar.gz"
        downloading.write_bytes(b"first updater still downloading")
        with (self.install / "tmp" / "update.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_update(success=False)
            self.assertEqual(downloading.read_bytes(), b"first updater still downloading")
        self.assertFalse(any(call[0] == "curl" for call in self.calls()))


if __name__ == "__main__":
    unittest.main()
