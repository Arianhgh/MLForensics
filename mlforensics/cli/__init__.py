"""Command-line interface and TOML configuration."""

from collections.abc import Sequence

from .config import Config, load_config, read_config


def build_parser():
    """Build the CLI parser without importing the command module eagerly."""
    from .main import build_parser as _build_parser

    return _build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    from .main import main as _main

    return _main(argv)


__all__ = ["Config", "build_parser", "load_config", "main", "read_config"]
