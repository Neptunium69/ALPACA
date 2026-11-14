"""Installer selection and failure behavior, without GitHub or pip mutations."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

SPEC = importlib.util.spec_from_file_location("alpaca_install", Path(__file__).resolve().parents[1] / "install.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        contexts = contextlib.ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(contextlib.redirect_stdout(self.output))
        contexts.enter_context(contextlib.redirect_stderr(self.output))

    def test_default_uses_latest_stable_release(self):
        payload = {"tag_name": "v0.3", "draft": False, "prerelease": False}
        with patch.object(installer, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as request, patch.object(installer.subprocess, "run") as run:
            self.assertEqual(installer.main(["--dry-run"]), 0)
        self.assertEqual(request.call_args.args[0].full_url, installer.LATEST_RELEASE)
        self.assertIn("@refs/tags/v0.3", self.output.getvalue())
        run.assert_not_called()

    def test_explicit_selections_do_not_query_latest(self):
        selections = [("--release", "v0.3", "refs/tags/v0.3"), ("--commit", "a" * 40, "a" * 40), ("--branch", "dev", "refs/heads/dev")]
        for flag, value, revision in selections:
            with self.subTest(flag=flag), patch.object(installer, "urlopen") as request:
                self.assertEqual(installer.main([flag, value, "--dry-run"]), 0)
                request.assert_not_called()
                self.assertIn("@" + revision, self.output.getvalue())

    def test_rejects_conflicting_or_malformed_selections(self):
        for arguments in (["--release", "v0.3", "--commit", "a" * 40], ["--commit", "abc123"], ["--release", "dev#subdirectory=elsewhere"], ["--branch", "../dev"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as error:
                installer.main(arguments)
            self.assertEqual(error.exception.code, 2)

    def test_no_fallback_to_dev_on_lookup_failure(self):
        for error in (URLError("offline"), HTTPError(installer.LATEST_RELEASE, 404, "Not Found", {}, None)):
            with self.subTest(error=error), patch.object(installer, "urlopen", side_effect=error), patch.object(installer.subprocess, "run") as run:
                self.assertEqual(installer.main([]), 1)
                run.assert_not_called()

    def test_rejects_nonstable_or_invalid_api_payload(self):
        for payload in ({"tag_name": "v0.4rc1", "prerelease": True}, {"tag_name": "v0.3", "draft": True}, {}, [], {"tag_name": "../dev"}):
            with self.subTest(payload=payload), patch.object(installer, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())), patch.object(installer.subprocess, "run") as run:
                self.assertEqual(installer.main([]), 1)
                run.assert_not_called()

    def fake_pip(self, command, check):
        self.assertTrue(check)
        if "wheel" in command:
            directory = Path(command[command.index("--wheel-dir") + 1])
            (directory / "alpaca_robotics-0.3.0-py3-none-any.whl").touch()

    def test_builds_once_and_replaces_same_version_without_reinstalling_dependencies(self):
        with patch.object(installer.subprocess, "run", side_effect=self.fake_pip) as run:
            self.assertEqual(installer.main(["--commit", "b" * 40, "--rover", "--no-build-isolation"]), 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertIn("--no-build-isolation", commands[0])
        self.assertTrue(commands[1][-1].endswith(".whl[rover]"))
        self.assertNotIn("--force-reinstall", commands[1])
        self.assertIn("--force-reinstall", commands[2])
        self.assertIn("--no-deps", commands[2])

    def test_no_deps_skips_dependency_resolution(self):
        with patch.object(installer.subprocess, "run", side_effect=self.fake_pip) as run:
            self.assertEqual(installer.main(["--release", "v0.3", "--no-deps"]), 0)
        self.assertEqual(run.call_count, 2)
        self.assertIn("--no-deps", run.call_args.args[0])

    def test_build_failure_never_installs_another_revision(self):
        error = subprocess.CalledProcessError(1, ["pip", "wheel"])
        with patch.object(installer.subprocess, "run", side_effect=error) as run:
            self.assertEqual(installer.main(["--release", "v0.1"]), 1)
        self.assertEqual(run.call_count, 1)
        self.assertIn("older snapshots may predate packaging", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
