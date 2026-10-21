"""IntentLab AI: calibrated intent classification with abstention on a BANKING77 subset."""

from importlib.metadata import version

# Single source of truth for the version is pyproject.toml; read it from installed metadata.
__version__: str = version("intentlab-ai")

__all__ = ["__version__"]
