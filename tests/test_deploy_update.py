"""Exercise the real simple updater with disposable releases and mocked Docker/network."""
from __future__ import annotations

import hashlib
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
MOCK = r'''#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys
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
if args[:2] == ["image", "inspect"]:
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
elif args and args[0] == "load":
    payload = json.loads(pathlib.Path(args[args.index("--input") + 1]).read_text())
    identity = "sha256:" + hashlib.sha256(payload["tag"].encode()).hexdigest()
    state["images"][identity] = payload
    state["tags"]["tgchatbot:" + payload["tag"] + "-" + payload["arch"]] = identity
elif args and args[0] == "tag":
    identity = image_id(args[1])
    if identity is None:
        sys.exit(1)
    state["tags"][args[2]] = identity
elif args and args[0] == "compose" and "up" in args:
    identity = image_id("tgchatbot:current")
    if identity is None or state["images"][identity]["tag"] == os.environ.get("FAIL_TAG"):
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
        shutil.copy2(REPO / ".github" / "release" / "update.sh", self.install / "update.sh")
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
                                MOCK_STATE=str(self.docker_state), TGCHATBOT_RELEASE_REPO="example/chatbot")
        for tag in ("v0.1.0", "v0.2.0"):
            self.make_release(tag)

    def make_release(self, tag):
        directory = self.assets / tag
        directory.mkdir(parents=True)
        for arch in ("amd64", "arm64"):
            payload = {"tag": tag, "arch": arch, "commit": COMMIT}
            (directory / f"tgchatbot-linux-{arch}.tar.gz").write_text(json.dumps(payload))
        with tarfile.open(directory / f"tgchatbot-deploy-{tag}.tar.gz", "w:gz") as archive:
            files = {"RELEASE_TAG": tag.encode(), "RELEASE_COMMIT": COMMIT.encode(),
                     ".env.example": b"TGBOT_TOKEN=\nOPENAI_API_KEY=\n"}
            for name in ("compose.yml", "update.sh"):
                files[name] = (REPO / ".github" / "release" / name).read_bytes()
            for name, contents in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))
        (directory / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted(directory.glob("*.tar.gz"))))

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

    def assert_simple_layout(self):
        self.assertFalse((self.install / "runtime").exists())
        self.assertFalse((self.install / "RELEASE_TAG").exists())
        self.assertFalse(list(self.install.rglob("release.env")))
        scratch = self.install / "tmp" / "update"
        self.assertEqual(list(scratch.iterdir()), [])

    def assert_data_preserved(self):
        self.assertEqual(self.state.read_bytes(), b"existing business state")
        self.assertEqual((self.install / ".env").read_text(), self.env_content)
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
        up = next(call for call in self.calls() if "up" in call)
        self.assertIn("--no-build", up)
        self.assertIn("--wait", up)
        self.assertIn("never", up)
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_bootstrap_creates_configuration_without_downloading_image(self):
        (self.install / ".env").unlink()
        (self.install / "update.sh").chmod(0o644)
        result = self.run_update()
        self.assertEqual((self.install / ".env").read_text(), "TGBOT_TOKEN=\nOPENAI_API_KEY=\n")
        self.assertTrue((self.install / "compose.yml").is_file())
        self.assertTrue(os.access(self.install / "update.sh", os.X_OK))
        self.assertFalse(any("load" in call or "up" in call for call in self.calls()))
        self.assertFalse(any("tgchatbot-linux-" in call[-1] for call in self.calls() if call[0] == "curl"))
        self.assert_simple_layout()

    def test_bad_checksum_never_loads_or_stops_services(self):
        for archive in (self.assets / "v0.2.0").glob("tgchatbot-linux-*.tar.gz"):
            archive.write_bytes(b"corrupted image")
        self.run_update("v0.2.0", success=False)
        self.assertFalse(any("load" in call or "stop" in call or "up" in call for call in self.calls()))
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_wrong_image_commit_never_stops_services(self):
        self.run_update("v0.2.0", success=False, MOCK_COMMIT="b" * 40)
        self.assertFalse(any("stop" in call or "up" in call for call in self.calls()))
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_failed_update_restores_current_image_and_compose_without_changing_data(self):
        self.run_update("v0.1.0")
        old_compose = (self.install / "compose.yml").read_text() + "\n# local prior file\n"
        (self.install / "compose.yml").write_text(old_compose)
        result = self.run_update("v0.2.0", success=False, FAIL_TAG="v0.2.0")
        self.assertEqual(self.image_tag("tgchatbot:current"), "v0.1.0")
        self.assertEqual((self.install / "compose.yml").read_text(), old_compose)
        self.assertEqual(len([call for call in self.calls() if "up" in call]), 3)
        self.assert_data_preserved()
        self.assert_simple_layout()

    def test_manual_rollback_swaps_images_without_network_or_deployment_metadata(self):
        self.run_update("v0.1.0")
        self.run_update("v0.2.0")
        downloads_before = len([call for call in self.calls() if call[0] == "curl"])
        self.run_update("rollback")
        self.assertEqual(self.image_tag("tgchatbot:current"), "v0.1.0")
        self.assertEqual(self.image_tag("tgchatbot:previous"), "v0.2.0")
        self.assertEqual(len([call for call in self.calls() if call[0] == "curl"]), downloads_before)
        self.assert_data_preserved()
        self.assert_simple_layout()

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
        downloading = scratch / "image.tar.gz"
        downloading.write_bytes(b"first updater still downloading")
        with (self.install / "tmp" / "update.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_update(success=False)
            self.assertEqual(downloading.read_bytes(), b"first updater still downloading")
        self.assertFalse(any(call[0] == "curl" for call in self.calls()))


if __name__ == "__main__":
    unittest.main()
