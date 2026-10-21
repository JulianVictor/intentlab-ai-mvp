"""Smoke tests for the installed intentlab_ai package."""

from importlib.metadata import version
from pathlib import Path

import intentlab_ai


def test_package_is_importable_from_src_layout() -> None:
    package_dir = Path(intentlab_ai.__file__).resolve().parent
    assert package_dir.name == "intentlab_ai"
    assert package_dir.parent.name == "src"


def test_version_matches_installed_distribution_metadata() -> None:
    assert intentlab_ai.__version__ == version("intentlab-ai")
