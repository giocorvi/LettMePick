""" Tests for the TMDB daily ID export inspector."""

import gzip
from pathlib import Path

import pytest

from src.inspect_tmdb_id_export import iter_records, parse_args


def test_iter_records_reads_gzip_json_lines(tmp_path: Path) -> None:
    """ Yield JSON records incrementally from a compressed export."""
    export_path = tmp_path / "movie_ids.json.gz"
    with gzip.open(export_path, "wt", encoding="utf-8") as export:
        export.write('{"id": 1, "adult": false}\n\n{"id": 2, "adult": true}\n')

    records = list(iter_records(export_path))

    assert records == [
        (1, {"id": 1, "adult": False}),
        (3, {"id": 2, "adult": True}),
    ]


def test_iter_records_reports_invalid_json_line(tmp_path: Path) -> None:
    """ Include the source line in malformed-export errors."""
    export_path = tmp_path / "movie_ids.json"
    export_path.write_text('{"id": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        list(iter_records(export_path))


def test_parse_args_accepts_local_export_path() -> None:
    """ Read the local export path from the command line."""
    args = parse_args(["data/movie_ids_08_30_2026.json.gz"])

    assert args.export_path == Path("data/movie_ids_08_30_2026.json.gz")
