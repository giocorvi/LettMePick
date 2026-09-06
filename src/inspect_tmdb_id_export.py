""" Open each record in a local TMDB daily ID export with ``ipdb``."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

import ipdb

"""
The local TMDB export contains one entry per movies, and the 'id' field is the TMDB ID (which is also present in the movie dictionary).
"""


def _open_export(export_path: Path) -> TextIO:
    """ Open a gzip-compressed or plain JSON Lines TMDB export.

    Args:
        export_path: Path to the local export file.

    Returns:
        A text stream for the export contents.
    """
    if not export_path.is_file():
        raise FileNotFoundError(f"TMDB export not found: {export_path}")
    if export_path.suffix == ".gz":
        return gzip.open(export_path, "rt", encoding="utf-8")
    return export_path.open("r", encoding="utf-8")


def iter_records(export_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """ Yield parsed TMDB records one at a time.

    Args:
        export_path: Path to a TMDB daily ID export.

    Yields:
        The one-based line number and decoded JSON record for each non-empty line.
    """
    with _open_export(export_path) as export:
        for line_number, line in enumerate(export, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {export_path}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected an object on line {line_number} of {export_path}"
                )
            yield line_number, record


def inspect_export(export_path: Path) -> None:
    """ Pause in ``ipdb`` for every record in a local TMDB export.

    Args:
        export_path: Path to a TMDB daily ID export.
    """
    for line_number, record in iter_records(export_path):
        print(f"Line {line_number}: id={record.get('id')}")
        print("Debugger locals include: export_path, line_number, and record")
        ipdb.set_trace()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """ Parse command-line arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "export_path",
        type=Path,
        help="local TMDB .json.gz or JSON Lines export path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """ Run the TMDB daily ID export inspector.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    inspect_export(args.export_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
