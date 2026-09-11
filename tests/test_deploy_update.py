"""Exercise the real updater with disposable releases and mocked Docker/network."""
from __future__ import annotations

import hashlib
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
import json, os, pathlib, sys
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
if args[:2] == ["image", "inspect"]:
    if "--format" in args:
        field = args[args.index("--format") + 1]
        if "image.version" in field:
            print(args[-1].split(":")[-1].rsplit("-", 1)[0])
        else:
            print(os.environ.get("MOCK_COMMIT", "a" * 40))
elif args and args[0] == "compose" and "--env-file" in args:
    env = pathlib.Path(args[args.index("--env-file") + 1]).read_text()
    if "up" in args and os.environ.get("FAIL_TAG") and ":" + os.environ["FAIL_TAG"] + "-" in env:
        sys.exit(1)
sys.exit(0)
'''


class ReleaseUpdaterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="deploy-test-")
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
        self.environment = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                                MOCK_LOG=str(self.log), MOCK_ASSETS=str(self.assets),
                                TGCHATBOT_RELEASE_REPO="example/chatbot", TGCHATBOT_UID="1000", TGCHATBOT_GID="1000")
        for tag in ("v0.1.0", "v0.2.0"):
            self.make_release(tag)

    def make_release(self, tag):
        directory = self.assets / tag
        directory.mkdir(parents=True)
        for arch in ("amd64", "arm64"):
            (directory / f"tgchatbot-linux-{arch}.tar.gz").write_bytes(b"mock docker save archive")
        with tarfile.open(directory / f"tgchatbot-deploy-{tag}.tar.gz", "w:gz") as archive:
            files = {"RELEASE_TAG": tag.encode(), "RELEASE_COMMIT": COMMIT.encode(), "README.md": b"deploy docs"}
            for name in ("compose.yml", "update.sh"):
                files[name] = (REPO / "deploy" / name).read_bytes()
            for name, contents in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))
        self.checksum(directory)

    def checksum(self, directory):
        (directory / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted(directory.glob("*.tar.gz"))))

    def run_update(self, target, *, success=True, **env):
        result = subprocess.run(["bash", str(self.install / "update.sh"), target],
                                cwd=self.install, env=dict(self.environment, **env),
                                text=True, capture_output=True, timeout=15)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def assert_data_preserved(self):
        self.assertEqual(self.state.read_bytes(), b"existing business state")
        self.assertEqual((self.install / ".env").read_text(), self.env_content)
        self.assertEqual((self.install / "data" / "tantivy_index" / "meta.json").read_text(), '{"preserve": true}')
        self.assertFalse((self.install / "SHOULD_NOT_EXIST").exists())

    def test_latest_resolves_once_then_downloads_exact_checksums_and_image(self):
        self.run_update("latest")
        self.assertEqual((self.install / "RELEASE_TAG").read_text(), "v0.2.0")
        urls = [call[-1] for call in self.calls() if call[0] == "curl"]
        self.assertTrue(urls[0].endswith("/releases/latest"))
        self.assertTrue(all("/releases/download/v0.2.0/" in url for url in urls[1:]))
        docker = [call for call in self.calls() if call[0] == "docker"]
        self.assertTrue(any(call[1] == "load" for call in docker))
        up = next(call for call in docker if "up" in call)
        self.assertIn("--no-build", up)
        self.assertIn("--wait", up)
        self.assertIn("never", up)
        self.assertNotIn("build", [call[1] for call in docker])
        self.assert_data_preserved()

    def test_bad_checksum_never_loads_or_stops_services(self):
        for archive in (self.assets / "v0.2.0").glob("tgchatbot-linux-*.tar.gz"):
            archive.write_bytes(b"corrupted image")
        self.run_update("v0.2.0", success=False)
        self.assertFalse(any("load" in call or "stop" in call or "up" in call for call in self.calls()))
        self.assert_data_preserved()

    def test_wrong_image_commit_never_stops_services(self):
        self.run_update("v0.2.0", success=False, MOCK_COMMIT="b" * 40)
        self.assertFalse(any("stop" in call or "up" in call for call in self.calls()))
        self.assert_data_preserved()

    def test_failed_update_restores_previous_code_without_changing_data(self):
        self.run_update("v0.1.0")
        result = self.run_update("v0.2.0", success=False, FAIL_TAG="v0.2.0")
        self.assertIn("Restored v0.1.0", result.stdout)
        self.assertEqual((self.install / "RELEASE_TAG").read_text(), "v0.1.0")
        self.assertFalse((self.install / "runtime" / "pending").exists())
        ups = [call for call in self.calls() if "up" in call]
        self.assertEqual(len(ups), 3)
        self.assertIn("v0.1.0", ups[-1][ups[-1].index("--env-file") + 1])
        self.assert_data_preserved()

    def test_manual_rollback_swaps_healthy_releases(self):
        self.run_update("v0.1.0")
        self.run_update("v0.2.0")
        self.run_update("rollback")
        self.assertEqual((self.install / "RELEASE_TAG").read_text(), "v0.1.0")
        self.assertEqual((self.install / "runtime" / "previous" / "RELEASE_TAG").read_text(), "v0.2.0")
        self.assert_data_preserved()

    def test_interrupted_update_requires_recovery_before_new_update(self):
        self.run_update("v0.1.0")
        active = (self.install / "runtime" / "current").resolve()
        (self.install / "runtime" / "pending").write_text(str(active))
        self.run_update("v0.2.0", success=False)
        self.run_update("recover")
        self.assertFalse((self.install / "runtime" / "pending").exists())
        self.assert_data_preserved()

    def test_failed_first_start_stops_services_and_retains_data(self):
        self.run_update("v0.1.0", success=False, FAIL_TAG="v0.1.0")
        self.assertFalse((self.install / "runtime" / "current").exists())
        self.assertFalse((self.install / "runtime" / "pending").exists())
        self.assertTrue(any("stop" in call for call in self.calls()))
        self.assert_data_preserved()

    def test_malformed_release_target_has_no_network_or_docker_side_effects(self):
        self.run_update("../../other", success=False)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
