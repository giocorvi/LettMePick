""" Inspect the TMDB v11 movie dataset."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "TMDB_movie_dataset_v11.csv"


@dataclass
class TMDBStats:
    column_names: List[str]
    rows: int
    distinct_genres: int
    distinct_keywords: int
    distinct_production_companies: int
    distinct_languages: int
    distinct_spoken_languages: int
    runtime_avg: float | None
    vote_average_mean: float | None
    vote_count_mean: float | None
    adult_ratio: float
    release_year_min: int | None
    release_year_max: int | None


def inspect_tmdb_dataset(data_path: Path = DATA_PATH) -> TMDBStats:
    """ Load and summarize the TMDB v11 dataset.

    Args:
        data_path: CSV file containing TMDB movie records.

    Returns:
        Aggregated statistics for the TMDB dataset.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Missing TMDB dataset at {data_path}")

    column_names: List[str] = []
    rows = 0

    genres: set[str] = set()
    keywords: set[str] = set()
    production_companies: set[str] = set()
    languages: set[str] = set()
    spoken_languages: set[str] = set()

    runtimes: List[float] = []
    vote_averages: List[float] = []
    vote_counts: List[float] = []
    adult_true = 0
    release_years: List[int] = []

    with data_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        column_names = reader.fieldnames or []

        for row in reader:
            rows += 1
            genres.update(_parse_list(row.get("genres")))
            keywords.update(_parse_list(row.get("keywords")))
            production_companies.update(_parse_list(row.get("production_companies")))
            spoken_languages.update(_parse_list(row.get("spoken_languages")))
            language = row.get("original_language")
            if language:
                languages.add(language.strip())

            _maybe_append_float(row.get("runtime"), runtimes)
            _maybe_append_float(row.get("vote_average"), vote_averages)
            _maybe_append_float(row.get("vote_count"), vote_counts)
            _maybe_append_year(row.get("release_date"), release_years)

            if str(row.get("adult", "")).strip().lower() == "true":
                adult_true += 1

    stats = TMDBStats(
        column_names=column_names,
        rows=rows,
        distinct_genres=len(genres - {""}),
        distinct_keywords=len(keywords - {""}),
        distinct_production_companies=len(production_companies - {""}),
        distinct_languages=len(languages - {""}),
        distinct_spoken_languages=len(spoken_languages - {""}),
        runtime_avg=mean(runtimes) if runtimes else None,
        vote_average_mean=mean(vote_averages) if vote_averages else None,
        vote_count_mean=mean(vote_counts) if vote_counts else None,
        adult_ratio=adult_true / rows if rows else 0.0,
        release_year_min=min(release_years) if release_years else None,
        release_year_max=max(release_years) if release_years else None,
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
    stripped = raw_value.strip()
    if not stripped:
        return ()
    try:
        value = json.loads(stripped)
        if isinstance(value, list):
            return [str(v).strip() for v in value]
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in stripped.split(",") if part.strip()]


def _maybe_append_float(raw_value: str | None, target: List[float]) -> None:
    """ Append a valid floating-point value to a collection.

    Args:
        raw_value: Candidate floating-point string.
        target: Collection updated when conversion succeeds.
    """
    try:
        if raw_value:
            target.append(float(raw_value))
    except ValueError:
        pass


def _maybe_append_year(raw_value: str | None, target: List[int]) -> None:
    """ Extract and append a valid year from a date string.

    Args:
        raw_value: Candidate date beginning with a four-digit year.
        target: Collection updated when conversion succeeds.
    """
    if not raw_value:
        return
    try:
        year = int(raw_value[:4])
        target.append(year)
    except ValueError:
        pass


def _print_stats(stats: TMDBStats) -> None:
    """ Print TMDB statistics in a readable format.

    Args:
        stats: Aggregated TMDB statistics to display.
    """
    print("TMDB v11 dataset summary")
    print("------------------------")
    print(f"Columns: {', '.join(stats.column_names)}")
    print(f"Rows: {stats.rows}")
    print(f"Distinct genres: {stats.distinct_genres}")
    print(f"Distinct keywords: {stats.distinct_keywords}")
    print(f"Distinct production companies: {stats.distinct_production_companies}")
    print(f"Original languages: {stats.distinct_languages}")
    print(f"Spoken languages: {stats.distinct_spoken_languages}")

    if stats.release_year_min is not None:
        print(
            "Release years: "
            f"{stats.release_year_min} - {stats.release_year_max}"
        )
    else:
        print("Release years: no valid dates")

    if stats.runtime_avg is not None:
        print(f"Average runtime: {stats.runtime_avg:.1f} minutes")
    else:
        print("Average runtime: not available")

    if stats.vote_average_mean is not None:
        print(f"Average TMDB rating: {stats.vote_average_mean:.2f}")
    else:
        print("Average TMDB rating: not available")

    if stats.vote_count_mean is not None:
        print(f"Average TMDB vote count: {stats.vote_count_mean:.1f}")
    else:
        print("Average TMDB vote count: not available")

    print(f"Adult-flagged titles: {stats.adult_ratio * 100:.2f}%")


if __name__ == "__main__":
    inspect_tmdb_dataset()
