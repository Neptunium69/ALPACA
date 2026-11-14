"""Install the latest ALPACA GitHub release, a release tag, or a commit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPOSITORY = "https://github.com/Neptunium69/ALPACA.git"
LATEST_RELEASE = "https://api.github.com/repos/Neptunium69/ALPACA/releases/latest"


def git_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value) or ".." in value or "//" in value or value.endswith(("/", ".", ".lock")):
        raise argparse.ArgumentTypeError("Use a Git tag or branch name without spaces or URL fragments.")
    return value


def commit_hash(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError("Use the full 40-character Git commit SHA.")
    return value.lower()


def latest_release() -> str:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "ALPACA-installer"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(LATEST_RELEASE, headers=headers), timeout=30) as response:
            release = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"GitHub latest-release lookup failed (HTTP {error.code}). Check repository access, set GITHUB_TOKEN if needed, or select --release/--commit explicitly.") from error
    except (URLError, OSError, ValueError) as error:
        raise RuntimeError("Could not read GitHub's latest release. Check the connection or select --release/--commit explicitly.") from error
    if not isinstance(release, dict) or release.get("draft") or release.get("prerelease") or not isinstance(release.get("tag_name"), str):
        raise RuntimeError("GitHub did not return a published stable release.")
    try:
        return git_name(release["tag_name"])
    except argparse.ArgumentTypeError as error:
        raise RuntimeError("GitHub returned an unsupported release tag.") from error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--release", type=git_name, metavar="TAG", help="Install this release tag instead of GitHub's latest release.")
    selection.add_argument("--commit", type=commit_hash, metavar="SHA", help="Install a specific full commit SHA.")
    selection.add_argument("--branch", type=git_name, metavar="NAME", help="Install a development branch, for example dev.")
    result.add_argument("--rover", action="store_true", help="Include Python perception/navigation dependencies; ROS is installed separately.")
    result.add_argument("--no-deps", action="store_true", help="Keep the environment's existing dependencies.")
    result.add_argument("--no-build-isolation", action="store_true", help="Use the environment's existing setuptools build backend.")
    result.add_argument("--dry-run", action="store_true", help="Print the selected source without building or installing it.")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.commit:
            revision = args.commit
        elif args.branch:
            revision = f"refs/heads/{args.branch}"
        else:
            revision = f"refs/tags/{args.release or latest_release()}"
        source = f"git+{REPOSITORY}@{quote(revision, safe='/')}"
        print(f"ALPACA source: {source}", flush=True)
        if args.dry_run:
            return 0

        pip = [sys.executable, "-m", "pip"]
        with TemporaryDirectory(prefix="alpaca-install-") as directory:
            build = [*pip, "wheel", "--no-deps", "--wheel-dir", directory]
            if args.no_build_isolation:
                build.append("--no-build-isolation")
            subprocess.run([*build, source], check=True)
            wheels = list(Path(directory).glob("alpaca_robotics-*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("The selected revision did not build exactly one alpaca-robotics wheel.")
            wheel = str(wheels[0])
            if not args.no_deps:
                requirement = wheel + ("[rover]" if args.rover else "")
                subprocess.run([*pip, "install", "--upgrade", requirement], check=True)
            # Different commits can share a version. Replace ALPACA itself without
            # force-reinstalling its dependencies (especially CUDA/PyTorch).
            subprocess.run([*pip, "install", "--force-reinstall", "--no-deps", wheel], check=True)
        return 0
    except subprocess.CalledProcessError as error:
        print("Installation failed. The selected revision must contain Python packaging metadata; older snapshots may predate packaging. See pip's error above.", file=sys.stderr)
        return error.returncode or 1
    except (RuntimeError, OSError) as error:
        print(f"Installation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
