"""Utilities for inspecting the exported ratings data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "ratings_export.csv"
SUBSET_PATH = BASE_DIR / "data" / "ratings_subset.csv"


@dataclass
class RatingsStats:
    column_names: List[str]
    rows: int
    distinct_users: int
    distinct_movies: int
    rating_min: float | None
    rating_max: float | None
    rating_avg: float | None


def inspect_ratings(
    data_path: Path = DATA_PATH,
    subset_path: Path = SUBSET_PATH,
    sample_users: int = 5,
    random_state: int | None = None,
) -> RatingsStats:
    """Load the ratings CSV, print summary info, and dump a subset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Missing ratings export at {data_path}")

    df = pd.read_csv(data_path)
    column_names = df.columns.tolist()

    if "rating_val" in df.columns:
        df["rating_val"] = pd.to_numeric(df["rating_val"], errors="coerce")

    rows = len(df)
    distinct_users = df["user_id"].nunique(dropna=True) if "user_id" in df else 0
    distinct_movies = df["movie_id"].nunique(dropna=True) if "movie_id" in df else 0
    rating_min = df["rating_val"].min() if "rating_val" in df else None
    rating_max = df["rating_val"].max() if "rating_val" in df else None
    rating_avg = df["rating_val"].mean() if "rating_val" in df else None

    subset_df, sampled_users = _sample_users(
        df, sample_size=sample_users, random_state=random_state
    )
    _write_subset(subset_path, subset_df)

    stats = RatingsStats(
        column_names=column_names,
        rows=rows,
        distinct_users=distinct_users,
        distinct_movies=distinct_movies,
        rating_min=rating_min,
        rating_max=rating_max,
        rating_avg=rating_avg,
    )

    _print_stats(stats, subset_path, subset_df, sampled_users)
    return stats


def _sample_users(
    df: pd.DataFrame, sample_size: int, random_state: int | None
) -> tuple[pd.DataFrame, List[str]]:
    if "user_id" not in df.columns:
        return df.head(0), []

    unique_users = df["user_id"].dropna().drop_duplicates()
    sample_size = min(sample_size, len(unique_users))
    sampled_users = (
        unique_users.sample(n=sample_size, random_state=random_state).tolist()
        if sample_size > 0
        else []
    )

    subset_df = df[df["user_id"].isin(sampled_users)].copy()
    return subset_df, sampled_users


def _write_subset(subset_path: Path, subset_df: pd.DataFrame) -> None:
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(subset_path, index=False)


def _print_stats(
    stats: RatingsStats,
    subset_path: Path,
    subset_df: pd.DataFrame,
    sampled_users: List[str],
) -> None:
    print("Ratings export summary")
    print("----------------------")
    print(f"Columns: {', '.join(stats.column_names)}")
    print(f"Rows: {stats.rows}")
    print(f"Distinct users: {stats.distinct_users}")
    print(f"Distinct movies: {stats.distinct_movies}")

    if stats.rating_min is not None:
        print(
            "Ratings: "
            f"min={stats.rating_min:.2f}, "
            f"max={stats.rating_max:.2f}, "
            f"avg={stats.rating_avg:.2f}"
        )
    else:
        print("Ratings: no numeric values found")

    print()
    if sampled_users:
        print(
            f"Wrote {len(subset_df)} rows for "
            f"{len(sampled_users)} random users to {subset_path}"
        )
        print(f"User IDs: {', '.join(sampled_users)}")
    else:
        print(f"No user data available to sample; created empty subset at {subset_path}")


if __name__ == "__main__":
    inspect_ratings()
