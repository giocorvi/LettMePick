import torch

from src.data.data import Movie
from src.data.prehash import (
    build_prehashed_bank,
    collate_prehashed_bank,
    prehash_movies,
)


def _movies() -> list[Movie]:
    """ Create representative movies for hashing tests.

    Args:
        None.

    Returns:
        Two movies with different metadata cardinalities.
    """
    return [
        Movie(
            id="movie-1",
            year=1999,
            genres=["drama"],
            actors=["Actor A", "Actor B"],
            directors=["Director A"],
        ),
        Movie(
            id="movie-2",
            year=2005,
            genres=["comedy"],
            actors=[],
            directors=["Director B"],
        ),
    ]


def test_prehash_is_deterministic_and_builds_masks() -> None:
    """ Verify deterministic hashing and correct padding masks.

    Args:
        None.

    Returns:
        None.
    """
    kwargs = {
        "num_id_buckets": 32,
        "num_actor_buckets": 16,
        "num_genre_buckets": 8,
        "num_director_buckets": 8,
    }

    first = prehash_movies(_movies(), **kwargs)
    second = prehash_movies(_movies(), **kwargs)

    assert first == second

    bank = build_prehashed_bank(first)
    assert set(bank) == {
        "id_idx",
        "year",
        "actors_idx",
        "actors_mask",
        "genres_idx",
        "genres_mask",
        "directors_idx",
        "directors_mask",
    }
    assert bank["actors_idx"].shape == (2, 2)
    torch.testing.assert_close(
        bank["actors_mask"],
        torch.tensor([[True, True], [False, False]]),
    )
    assert bank["id_idx"].dtype == torch.long
    assert bank["year"].dtype == torch.float32


def test_collate_selects_movies_in_batch_order() -> None:
    """ Verify collation preserves requested movie order.

    Args:
        None.

    Returns:
        None.
    """
    prehashed = prehash_movies(
        _movies(),
        num_id_buckets=32,
        num_actor_buckets=16,
        num_genre_buckets=8,
        num_director_buckets=8,
    )
    bank = build_prehashed_bank(prehashed)

    batch = collate_prehashed_bank([[1, 0], [0, 1]], bank)

    assert batch["id_idx"].shape == (4,)
    torch.testing.assert_close(
        batch["year"],
        torch.tensor([2005.0, 1999.0, 1999.0, 2005.0]),
    )
