"""Prepare a v0.3 source snapshot without changing the development checkout."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil


def prepare(destination: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    destination = destination.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Choose an output directory outside the development repository.")
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")

    metadata = (root / "pyproject.toml").read_text()
    for before, after in [('requires-python = ">=3.11"', 'requires-python = ">=3.10"'), ('dependencies = ["torch>=2.4"]', 'dependencies = []')]:
        if metadata.count(before) != 1:
            raise ValueError(f"Review the v0.3 metadata transformation: {before}")
        metadata = metadata.replace(before, after)
    version, changes = re.subn(r'^__version__ = ".*"$', '__version__ = "0.3.0"', (root / "src/alpaca/__init__.py").read_text(), flags=re.MULTILINE)
    if changes != 1:
        raise ValueError("Could not identify the package version.")

    destination.mkdir(parents=True)
    for name in ("README.md", "LICENSE", "MANIFEST.in", "install.py", "requirements-ros.txt"):
        shutil.copy2(root / name, destination / name)
    for name in ("src", "ros1"):
        shutil.copytree(root / name, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "moc", "legacy"))
    (destination / "pyproject.toml").write_text(metadata)
    (destination / "src/alpaca/__init__.py").write_text(version)
    readme = (destination / "README.md").read_text().replace("Use Python 3.11 or newer for development installs", "Use Python 3.10 or newer for this release")
    (destination / "README.md").write_text(readme)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the release source snapshot.")
    args = parser.parse_args()
    try:
        destination = prepare(args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Could not prepare release: {error}\n")
    print(f"Prepared v0.3 source at {destination}")
    print("Build and tag this snapshot for v0.3; the development checkout is unchanged.")


if __name__ == "__main__":
    main()
