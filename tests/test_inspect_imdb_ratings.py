""" Tests for the IMDb page inspection script."""

from pathlib import Path

import pytest
import torch

from src.inspect_imdb_ratings import (
    fetch_page,
    load_movie,
    load_page,
    normalize_imdb_url,
    parse_args,
)


def test_load_movie_matches_non_string_dictionary_key(tmp_path: Path) -> None:
    """ Load an entry whose serialized key is not a string."""
    data_path = tmp_path / "movies.pt"
    torch.save({629: {"movie_title": "The Usual Suspects"}}, data_path)

    key, movie = load_movie(data_path, "629")

    assert key == 629
    assert movie["movie_title"] == "The Usual Suspects"


def test_load_movie_rejects_unknown_entry(tmp_path: Path) -> None:
    """ Report a missing requested dictionary key."""
    data_path = tmp_path / "movies.pt"
    torch.save({"known": {}}, data_path)

    with pytest.raises(KeyError, match="missing"):
        load_movie(data_path, "missing")


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    [
        (
            "http://www.imdb.com/title/tt0114814/maindetails",
            "http://www.imdb.com/title/tt0114814",
        ),
        (
            "[https://www.imdb.com/title/tt0114814/](https://www.imdb.com/title/tt0114814/)",
            "https://www.imdb.com/title/tt0114814/",
        ),
    ],
)
def test_normalize_imdb_url(raw_url: str, expected: str) -> None:
    """ Accept both plain and Markdown-formatted IMDb links."""
    assert normalize_imdb_url(raw_url) == expected


def test_parse_args_preserves_entry_and_cache_path() -> None:
    """ Read the movie dictionary key and cache path from the command line."""
    args = parse_args(["--entry", "film-key"])

    assert args.entry == "film-key"
    assert args.save_path is None
    assert args.load_path is None


def test_fetch_page_caches_and_parses_response_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ Cache and parse a successful HTTP response without full text decoding."""
    class Response:
        status_code = 200
        url = "https://example.test/movie"
        headers = {"content-type": "text/html", "content-encoding": None}
        encoding = "utf-8"
        content = b"<html><title>Example movie</title></html>"

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

    response = Response()
    requested: dict[str, object] = {}

    def fake_get(*args: object, **kwargs: object) -> Response:
        requested["args"] = args
        requested["kwargs"] = kwargs
        return response

    monkeypatch.setattr("src.inspect_imdb_ratings.requests.get", fake_get)

    save_path = tmp_path / "pages" / "example.html"
    soup = fetch_page("https://example.test/movie", timeout=12.5, save_path=save_path)

    assert soup.title is not None
    assert soup.title.string == "Example movie"
    assert save_path.read_bytes() == response.content
    assert requested["args"] == ("https://example.test/movie",)
    assert requested["kwargs"] is not None


def test_load_page_parses_cached_bytes(tmp_path: Path) -> None:
    """ Parse cached page bytes without making an HTTP request."""
    load_path = tmp_path / "cached.html"
    load_path.write_bytes(b"<html><title>Cached movie</title></html>")

    soup = load_page(load_path)

    assert soup.title is not None
    assert soup.title.string == "Cached movie"
