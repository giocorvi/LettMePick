""" Inspect metadata for exported movie data."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import List, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "movie_data.csv"


@dataclass
class MovieStats:
    column_names: List[str]
    rows: int
    distinct_genres: int
    distinct_languages: int
    release_year_min: int | None
    release_year_max: int | None
    runtime_avg: float | None
    vote_average_mean: float | None
    vote_count_total: int


def inspect_movies(data_path: Path = DATA_PATH) -> MovieStats:
    """ Load and summarize exported movie metadata.

    Args:
        data_path: CSV file containing exported movie metadata.

    Returns:
        Aggregated statistics for the movie dataset.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Missing movie data at {data_path}")

    column_names: List[str] = []
    rows = 0
    genres: set[str] = set()
    languages: set[str] = set()
    release_years: List[int] = []
    runtimes: List[float] = []
    vote_averages: List[float] = []
    vote_count_total = 0

    with data_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        column_names = reader.fieldnames or []
        for row in reader:
            rows += 1
            genres.update(_parse_list(row.get("genres")))
            languages.update(_parse_list(row.get("spoken_languages")))

            _maybe_append_int(row.get("year_released"), release_years)
            _maybe_append_float(row.get("runtime"), runtimes)
            _maybe_append_float(row.get("vote_average"), vote_averages)
            vote_count_total += _parse_int(row.get("vote_count"))

    stats = MovieStats(
        column_names=column_names,
        rows=rows,
        distinct_genres=len(genres - {""}),
        distinct_languages=len(languages - {""}),
        release_year_min=min(release_years) if release_years else None,
        release_year_max=max(release_years) if release_years else None,
        runtime_avg=mean(runtimes) if runtimes else None,
        vote_average_mean=mean(vote_averages) if vote_averages else None,
        vote_count_total=vote_count_total,
    )

    _print_stats(stats)
    return stats


def _parse_list(raw_value: str | None) -> Sequence[str]:
    """ Parse a JSON or comma-separated list value.

    Args:
        raw_value: Serialized list value, or ``None``.

    Returns:
        Parsed and trimmed string values.
    """
    if not raw_value:
        return ()
    try:
        value = json.loads(raw_value)
        if isinstance(value, list):
            return [str(v).strip() for v in value]
    except json.JSONDecodeError:
        pass
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def _maybe_append_int(raw_value: str | None, target: List[int]) -> None:
    """ Append a valid integer value to a collection.

    Args:
        raw_value: Candidate integer string.
        target: Collection updated when conversion succeeds.

    Returns:
        None.
    """
    try:
        if raw_value:
            target.append(int(raw_value))
    except ValueError:
        pass


def _maybe_append_float(raw_value: str | None, target: List[float]) -> None:
    """ Append a valid floating-point value to a collection.

    Args:
        raw_value: Candidate floating-point string.
        target: Collection updated when conversion succeeds.

    Returns:
        None.
    """
    try:
        if raw_value:
            target.append(float(raw_value))
    except ValueError:
        pass


def _parse_int(raw_value: str | None) -> int:
    """ Convert a value to an integer with a zero fallback.

    Args:
        raw_value: Candidate integer string.

    Returns:
        The parsed integer, or zero when parsing fails.
    """
    try:
        return int(raw_value) if raw_value else 0
    except ValueError:
        return 0


def _print_stats(stats: MovieStats) -> None:
    """ Print movie statistics in a readable format.

    Args:
        stats: Aggregated movie statistics to display.

    Returns:
        None.
    """
    print("Movie data summary")
    print("------------------")
    print(f"Columns: {', '.join(stats.column_names)}")
    print(f"Rows: {stats.rows}")
    print(f"Distinct genres: {stats.distinct_genres}")
    print(f"Distinct spoken languages: {stats.distinct_languages}")

    if stats.release_year_min is not None:
        print(
            "Release years: "
            f"{stats.release_year_min} - {stats.release_year_max}"
        )
    else:
        print("Release years: no numeric values found")

    if stats.runtime_avg is not None:
        print(f"Average runtime: {stats.runtime_avg:.1f} minutes")
    else:
        print("Average runtime: no numeric values found")

    if stats.vote_average_mean is not None:
        print(f"Average vote score: {stats.vote_average_mean:.2f}")
    else:
        print("Average vote score: no numeric values found")

    print(f"Total TMDB vote count: {stats.vote_count_total}")


if __name__ == "__main__":
    inspect_movies()
