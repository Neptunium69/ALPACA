"""Verify release preparation does not remove development-only source."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_source", ROOT / "scripts/prepare_v03_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseSourceTests(unittest.TestCase):
    def test_v03_snapshot_excludes_moc_but_dev_keeps_it(self):
        source = ROOT / "src/alpaca/moc/moc.py"
        original = source.read_bytes()
        with TemporaryDirectory() as directory:
            output = release.prepare(Path(directory) / "v0.3")
            self.assertFalse((output / "src/alpaca/moc").exists())
            self.assertIn('__version__ = "0.3.0"', (output / "src/alpaca/__init__.py").read_text())
            metadata = (output / "pyproject.toml").read_text()
            self.assertIn('requires-python = ">=3.10"', metadata)
            self.assertIn('dependencies = []', metadata)
            self.assertTrue((output / "src/alpaca/resources/mission_prompt.txt").is_file())
        self.assertEqual(source.read_bytes(), original)
        self.assertNotIn('"alpaca.moc"', (ROOT / "pyproject.toml").read_text())
        self.assertNotIn('prune src/alpaca/moc', (ROOT / "MANIFEST.in").read_text())

    def test_refuses_to_overwrite_existing_output(self):
        with TemporaryDirectory() as directory, self.assertRaises(FileExistsError):
            release.prepare(Path(directory))

    def test_refuses_output_inside_development_source(self):
        with self.assertRaises(ValueError):
            release.prepare(ROOT / "release")


if __name__ == "__main__":
    unittest.main()
