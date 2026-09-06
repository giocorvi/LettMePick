""" Tests for local filesystem context persistence."""

from pathlib import Path

from src.ui.context_store import ContextStore


def _movies(rating: float = 4.5) -> list[dict[str, object]]:
    """ Build one canonical rated movie record.

    Args:
        rating: Five-star rating to store.

    Returns:
        One serializable movie record.
    """
    return [
        {
            "id": "m1",
            "title": "Alien",
            "year": 1979,
            "genres": ["horror"],
            "rating": rating,
        }
    ]


def test_contexts_persist_update_and_delete(tmp_path: Path) -> None:
    """ Verify contexts survive store instances and update by model and name.

    Args:
        tmp_path: Temporary persistence directory.
    """
    directory = tmp_path / "saved_contexts"
    first_store = ContextStore(directory)
    first = first_store.save_context(
        "Sunday matinee", "first-cut", "First Cut", _movies()
    )

    second_store = ContextStore(directory)
    loaded = second_store.list_contexts()
    updated = second_store.save_context(
        "sunday MATINEE", "first-cut", "First Cut", _movies(5.0)
    )

    assert first_store.path == directory / "contexts.json"
    assert loaded == [first]
    assert updated["id"] == first["id"]
    assert updated["movies"][0]["rating"] == 5.0
    assert len(second_store.list_contexts()) == 1

    second_store.delete_context(str(first["id"]))

    assert ContextStore(directory).list_contexts() == []
