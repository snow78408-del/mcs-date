"""Command line entry point for ``python -m mcs_date``."""

from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mcs-date",
        description="Experiment record and data versioning utilities for MCS data.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    parser.parse_args(argv)
    print(f"mcs-date {__version__} - scaffold ready. Add your commands here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
