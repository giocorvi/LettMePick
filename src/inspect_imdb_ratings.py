""" Fetch one movie's IMDb page and open an interactive debugger."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ipdb
import requests
import torch
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "checkpoints" / "movies-data-dict_complete.pt"
MARKDOWN_LINK = re.compile(r"^\[(https?://[^]]+)]\([^)]+\)$")


def load_movie(data_path: Path, entry: str) -> tuple[Any, dict[str, Any]]:
    """ Load a movie selected by its dictionary key.

    Args:
        data_path: Path to the serialized movie dictionary.
        entry: String representation of the desired dictionary key.

    Returns:
        The original dictionary key and its movie metadata.
    """
    if not data_path.is_file():
        raise FileNotFoundError(f"Movie dictionary not found: {data_path}")

    movies = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(movies, Mapping):
        raise TypeError(f"Expected a movie dictionary, got {type(movies).__name__}")

    matching_keys = [key for key in movies if str(key) == entry]
    if not matching_keys:
        raise KeyError(f"Movie entry {entry!r} was not found in {data_path}")
    if len(matching_keys) > 1:
        raise KeyError(f"Movie entry {entry!r} matches more than one dictionary key")

    key = matching_keys[0]
    movie = movies[key]
    if not isinstance(movie, dict):
        raise TypeError(
            f"Expected metadata for entry {entry!r} to be a dictionary, "
            f"got {type(movie).__name__}"
        )
    return key, movie


def normalize_imdb_url(raw_url: Any) -> str:
    """ Extract an IMDb URL from a plain URL or Markdown link.

    Args:
        raw_url: Value stored in the movie's ``imdb_link`` field.

    Returns:
        A URL suitable for an HTTP request.
    """
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("The selected movie has no usable imdb_link")

    url = raw_url.strip()
    markdown_match = MARKDOWN_LINK.fullmatch(url)
    if markdown_match:
        url = markdown_match.group(1)
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Unsupported IMDb URL: {url!r}")
    return url.removesuffix("/maindetails")


def fetch_page(
    url: str, timeout: float, save_path: Path | None = None
) -> BeautifulSoup:
    """ Fetch, optionally cache, and parse an HTML page.

    Args:
        url: Page URL.
        timeout: Maximum number of seconds to wait for the response.
        save_path: Optional destination for the raw response bytes.

    Returns:
        Parsed page contents.
    """
    headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        ),
    }
    with requests.get(url, headers=headers, timeout=timeout) as response:
        response.raise_for_status()
        content = response.content
        preview = content[:500].decode(response.encoding or "utf-8", errors="replace")

        print("status:", response.status_code)
        print("url:", response.url)
        print("content-type:", response.headers.get("content-type"))
        print("content-encoding:", response.headers.get("content-encoding"))
        print("bytes:", len(content))
        print("text:", repr(preview))

    if not content:
        raise RuntimeError(f"The page response was empty: url={url}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(content)
        print(f"Saved page bytes to {save_path}")


    return BeautifulSoup(content, "html.parser")


def load_page(load_path: Path) -> BeautifulSoup:
    """ Load and parse previously cached page bytes.

    Args:
        load_path: Path containing cached HTML bytes.

    Returns:
        Parsed page contents.
    """
    if not load_path.is_file():
        raise FileNotFoundError(f"Cached page not found: {load_path}")

    content = load_path.read_bytes()
    if not content:
        raise RuntimeError(f"Cached page is empty: {load_path}")
    return BeautifulSoup(content, "html.parser")


def inspect_entry(
    data_path: Path,
    entry: str,
    timeout: float = 30.0,
    use_letterboxd_url: bool = False,
    save_path: Path | None = None,
) -> None:
    """ Fetch one movie's IMDb page and pause inside ``ipdb``.

    Args:
        data_path: Path to the serialized movie dictionary.
        entry: String representation of the desired dictionary key.
        timeout: Maximum number of seconds to wait for IMDb.
        use_letterboxd_url: If True, use the Letterboxd URL instead of IMDb.
        save_path: Optional destination for the fetched page bytes.
    """
    if use_letterboxd_url:
        url = f"https://letterboxd.com/film/{entry}"
        movie_title = entry
    else:
        movie_key, movie = load_movie(data_path, entry)
        url = normalize_imdb_url(movie.get("imdb_link"))
        movie_title = movie.get("movie_title", movie_key)

    soup = fetch_page(url, timeout, save_path)

    print(f"Loaded {movie_title!r} from {url}")
    print("Debugger locals include: entry, url, movie_title, and soup")
    ipdb.set_trace()


def inspect_cached_page(load_path: Path) -> None:
    """ Open a cached page in ``ipdb`` without making a network request.

    Args:
        load_path: Path containing cached HTML bytes.
    """
    soup = load_page(load_path)
    print(f"Loaded cached page from {load_path}")
    print("Debugger locals include: load_path and soup")
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
        "--entry",
        help="dictionary key of the movie to inspect (for example, 629)",
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--letterboxd",
        action="store_true",
        help="Use the Letterboxd stats URL instead of the stored IMDb URL",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--save-path",
        type=Path,
        help="write fetched page bytes to this path",
    )
    cache_group.add_argument(
        "--load-path",
        type=Path,
        help="load cached page bytes from this path without using the network",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """ Run the IMDb page inspector.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    if args.load_path is not None:
        inspect_cached_page(args.load_path)
        return 0
    if args.entry is None:
        raise ValueError("--entry is required unless --load-path is provided")

    inspect_entry(
        args.data_path,
        args.entry,
        args.timeout,
        args.letterboxd,
        args.save_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
