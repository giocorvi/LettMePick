""" Integration tests for the local model-testing API."""

from dataclasses import asdict
from pathlib import Path

import torch
import pytest
from fastapi.testclient import TestClient

from src.ui.app import create_app
from src.utils.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from src.utils.train import build_model


def _write_checkpoint(tmp_path: Path, name: str = "model.pt", feature_size: int = 4) -> Path:
    """ Create a tiny valid inference checkpoint and movie dataset.

    Args:
        tmp_path: Temporary test directory.
        name: Checkpoint filename.
        feature_size: Embedded model feature width.

    Returns:
        Written checkpoint path.
    """
    movies = {
        "m1": {"title": "Alien", "year_released": 1979, "letterboxd_genres": ["Horror"], "letterboxd_rating_count": 2_000_000, "actors": ["A"], "director": ["D"]},
        "m2": {"title": "Aliens", "year_released": 1986, "letterboxd_genres": ["Action"], "letterboxd_rating_count": 99_999, "actors": ["B"], "director": ["D"]},
        "m3": {"title": "The Alienist", "year_released": 1994, "letterboxd_genres": ["Drama"], "letterboxd_rating_count": 100_000, "actors": ["C"], "director": ["E"]},
        "m4": {"title": "Arrival", "year_released": 2016, "letterboxd_genres": ["Science Fiction"], "letterboxd_rating_count": 1_000_000, "actors": ["D"], "director": ["F"]},
        "m5": {"movie_title": "Solaris", "year_released": 1972, "letterboxd_genres": ["Drama"], "actors": [], "director": ["G"]},
    }
    movies_path = tmp_path / f"{name}-movies.pt"
    torch.save(movies, movies_path)
    config = ExperimentConfig(
        data=DataConfig(
            ratings_path=tmp_path / "unused-ratings.pt",
            movies_path=movies_path,
            test_ratio=0.2,
            val_ratio=0.2,
            num_id_buckets=16,
            num_actor_buckets=16,
            num_genre_buckets=8,
            num_director_buckets=8,
        ),
        model=ModelConfig(
            feature_size=feature_size,
            model_embed_dim=8,
            num_attention_heads=2,
            num_movie_attention_blocks=1,
            num_self_attention_blocks=1,
            num_cross_attention_blocks=1,
        ),
        training=TrainingConfig(
            epochs=1,
            steps_per_epoch=1,
            validation_steps=1,
            test_steps=1,
            min_context_size=1,
            max_context_size=2,
            min_target_size=2,
            max_target_size=2,
            progress=False,
        ),
        runtime=RuntimeConfig(device="cpu"),
    )
    model = build_model(config)
    checkpoint = tmp_path / name
    torch.save({"config": asdict(config), "best_model_state_dict": model.state_dict()}, checkpoint)
    return checkpoint


def _client(tmp_path: Path) -> TestClient:
    """ Create a client backed by two trusted registry entries.

    Args:
        tmp_path: Temporary test directory.

    Returns:
        Configured synchronous API client.
    """
    first = _write_checkpoint(tmp_path, "first.pt", feature_size=4)
    second = _write_checkpoint(tmp_path, "second.pt", feature_size=6)
    inspection = tmp_path / "inspection.toml"
    inspection.write_text(
        "\n".join(
            [
                "[data]",
                'ratings_path = "unused-ratings.pt"',
                'movies_path = "first.pt-movies.pt"',
                "test_ratio = 0.2",
                "val_ratio = 0.2",
                "num_id_buckets = 16",
                "num_actor_buckets = 16",
                "num_genre_buckets = 8",
                "num_director_buckets = 8",
                "",
                "[model]",
                "feature_size = 4",
                "model_embed_dim = 8",
                "num_attention_heads = 2",
                "num_movie_attention_blocks = 1",
                "num_self_attention_blocks = 1",
                "num_cross_attention_blocks = 1",
                "",
                "[training]",
                "batch_size = 7",
                "",
                "[runtime]",
                'device = "cpu"',
            ]
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "models.toml"
    registry.write_text(
        "\n".join(
            [
                "[[models]]",
                'name = "First Cut"',
                f'path = "{first.name}"',
                f'config = "{inspection.name}"',
                'description = "Validation selection"',
                "",
                "[[models]]",
                'name = "Second Cut"',
                f'path = "{second.name}"',
            ]
        ),
        encoding="utf-8",
    )
    return TestClient(create_app(registry))


def _context() -> list[dict[str, object]]:
    """ Return one valid half-star context payload."""
    return [{"movie_id": "m1", "rating": 4.5}]


def test_registry_config_overrides_checkpoint_config(tmp_path: Path) -> None:
    """ Verify a registry config is inspectable and overrides checkpoint values.

    Args:
        tmp_path: Temporary test directory.
    """
    client = _client(tmp_path)

    listed = client.get("/api/models").json()["models"]
    inspected = client.get("/api/models/first-cut/config")
    loaded = client.post("/api/models/first-cut/load")

    assert listed[0]["has_config"] is True
    assert listed[0]["loading"] is False
    assert "batch_size = 7" in inspected.json()["content"]
    assert loaded.status_code == 200
    assert loaded.json()["architecture"]["feature_size"] == 4
    assert client.app.state.runtime.active.checkpoint_config.training.batch_size == 7


def test_model_reload_uses_prepared_movie_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ Verify later loads do not rebuild an unchanged movie bank.

    Args:
        tmp_path: Temporary test directory.
        monkeypatch: Pytest attribute patch helper.
    """
    client = _client(tmp_path)
    runtime = client.app.state.runtime
    first = runtime.load_model("first-cut")
    cache_path = runtime._movie_cache_path(runtime.active.checkpoint_config)

    def reject_preparation(*args: object, **kwargs: object) -> None:
        """ Fail if cached movie data is unexpectedly rebuilt.

        Args:
            args: Ignored positional arguments.
            kwargs: Ignored keyword arguments.
        """
        raise AssertionError("movie data should have been loaded from cache")

    monkeypatch.setattr("src.ui.runtime.prepare_dataset", reject_preparation)
    second = runtime.load_model("first-cut")

    assert first["id"] == "first-cut"
    assert cache_path.is_file()
    assert second["id"] == "first-cut"


def test_version_two_movie_cache_migrates_without_rehashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ Verify cache migration reparses metadata while preserving the hashed bank.

    Args:
        tmp_path: Temporary test directory.
        monkeypatch: Pytest attribute patch helper.
    """
    client = _client(tmp_path)
    runtime = client.app.state.runtime
    runtime.load_model("first-cut")
    assert runtime.active is not None
    cache_path = runtime._movie_cache_path(runtime.active.checkpoint_config)
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    cached["version"] = 2
    torch.save(cached, cache_path)

    def reject_preparation(*args: object, **kwargs: object) -> None:
        """ Fail if migration attempts to rebuild hashed tensors.

        Args:
            args: Ignored positional arguments.
            kwargs: Ignored keyword arguments.
        """
        raise AssertionError("version-2 migration must reuse the hashed bank")

    monkeypatch.setattr("src.ui.runtime.prepare_dataset", reject_preparation)
    runtime.load_model("first-cut")
    migrated = torch.load(cache_path, map_location="cpu", weights_only=False)

    assert migrated["version"] == 3
    assert runtime.active is not None
    assert runtime.active.catalog_by_id["m4"].letterboxd_rating_count == 1_000_000


def test_search_and_manual_predictions_are_ranked(tmp_path: Path) -> None:
    """ Verify prefix-first title search and decreasing manual prediction scores.

    Args:
        tmp_path: Temporary test directory.
    """
    client = _client(tmp_path)
    client.post("/api/models/first-cut/load")

    search = client.get("/api/movies", params={"query": "alien"}).json()["movies"]
    response = client.post(
        "/api/predictions/manual",
        json={"context": _context(), "query_movie_ids": ["m2", "m3", "m4"]},
    )

    assert [movie["title"] for movie in search] == ["Alien", "Aliens", "The Alienist"]
    assert response.status_code == 200
    scores = [movie["score"] for movie in response.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_auto_run_samples_every_candidate_once(tmp_path: Path) -> None:
    """ Verify auto mode accumulates unique movies until catalog exhaustion.

    Args:
        tmp_path: Temporary test directory.
    """
    client = _client(tmp_path)
    client.post("/api/models/first-cut/load")
    payload = client.post(
        "/api/auto-runs", json={"context": _context(), "batch_size": 2}
    ).json()

    while not payload["complete"]:
        payload = client.post(f"/api/auto-runs/{payload['run_id']}/next").json()

    result_ids = [movie["id"] for movie in payload["results"]]
    assert payload["sampled"] == 4
    assert len(result_ids) == len(set(result_ids)) == 4
    assert "m1" not in result_ids


def test_popularity_facets_and_filtered_auto_run(tmp_path: Path) -> None:
    """ Verify popularity tiers are authoritative and auto candidates meet the minimum.

    Args:
        tmp_path: Temporary test directory.
    """
    client = _client(tmp_path)
    client.post("/api/models/first-cut/load")

    facets = client.get("/api/movie-facets").json()
    response = client.post(
        "/api/auto-runs",
        json={
            "context": _context(),
            "batch_size": 8,
            "min_rating_count": 100_000,
        },
    )

    assert facets["popularity_levels"] == [
        {"id": "any", "label": "Any", "min_rating_count": 0},
        {"id": "established", "label": "Established", "min_rating_count": 1_000},
        {"id": "popular", "label": "Popular", "min_rating_count": 100_000},
        {"id": "blockbuster", "label": "Blockbuster", "min_rating_count": 1_000_000},
    ]
    assert response.status_code == 200
    assert {movie["id"] for movie in response.json()["results"]} == {"m3", "m4"}


def test_auto_run_rejects_negative_popularity_threshold(tmp_path: Path) -> None:
    """ Verify API validation rejects nonsensical minimum rating counts.

    Args:
        tmp_path: Temporary test directory.
    """
    client = _client(tmp_path)
    response = client.post(
        "/api/auto-runs",
        json={"context": _context(), "batch_size": 2, "min_rating_count": -1},
    )

    assert response.status_code == 422


def test_loading_another_model_invalidates_auto_run(tmp_path: Path) -> None:
    """ Verify model switching releases catalog-specific cached context.

    Args:
        tmp_path: Temporary test directory.
    """
    client = _client(tmp_path)
    client.post("/api/models/first-cut/load")
    run = client.post("/api/auto-runs", json={"context": _context(), "batch_size": 1}).json()

    loaded = client.post("/api/models/second-cut/load")
    old_run = client.post(f"/api/auto-runs/{run['run_id']}/next")

    assert loaded.json()["architecture"]["feature_size"] == 6
    assert old_run.status_code == 404


def test_runtime_loads_a_module_stored_directly_in_checkpoint(tmp_path: Path) -> None:
    """ Verify torch.load may return a configured LettMePick module directly.

    Args:
        tmp_path: Temporary test directory.
    """
    checkpoint = _write_checkpoint(tmp_path, "state.pt")
    state_registry = tmp_path / "state-models.toml"
    state_registry.write_text(
        f'[[models]]\nname = "State"\npath = "{checkpoint.name}"\n',
        encoding="utf-8",
    )
    runtime = create_app(state_registry).state.runtime
    runtime.load_model("state")
    assert runtime.active is not None
    config = runtime.active.checkpoint_config
    module = build_model(config)
    module.experiment_config = config
    module_path = tmp_path / "module.pt"
    torch.save(module, module_path)
    module_registry = tmp_path / "module-models.toml"
    module_registry.write_text(
        f'[[models]]\nname = "Module"\npath = "{module_path.name}"\n',
        encoding="utf-8",
    )

    runtime = create_app(module_registry).state.runtime
    loaded = runtime.load_model("module")

    assert loaded["id"] == "module"
    assert runtime.active is not None
    assert isinstance(runtime.active.model, type(module))
    assert runtime.active.model.experiment_config == config


def test_direct_module_checkpoint_requires_embedded_experiment_config(tmp_path: Path) -> None:
    """ Verify a bare saved module cannot guess its movie data.

    Args:
        tmp_path: Temporary test directory.
    """
    checkpoint = _write_checkpoint(tmp_path, "state.pt")
    state_registry = tmp_path / "state-models.toml"
    state_registry.write_text(
        f'[[models]]\nname = "State"\npath = "{checkpoint.name}"\n',
        encoding="utf-8",
    )
    runtime = create_app(state_registry).state.runtime
    runtime.load_model("state")
    assert runtime.active is not None
    module = build_model(runtime.active.checkpoint_config)
    module_path = tmp_path / "bare-module.pt"
    torch.save(module, module_path)
    registry = tmp_path / "module-models.toml"
    registry.write_text(
        f'[[models]]\nname = "Module"\npath = "{module_path.name}"\n',
        encoding="utf-8",
    )
    runtime = create_app(registry).state.runtime

    try:
        runtime.load_model("module")
    except ValueError as error:
        assert "experiment_config" in str(error)
    else:
        raise AssertionError("Bare direct-module checkpoints must be rejected.")


def test_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    """ Verify inference requires a complete checkpoint-embedded configuration.

    Args:
        tmp_path: Temporary test directory.
    """
    checkpoint = tmp_path / "broken.pt"
    torch.save({"model_state_dict": {}}, checkpoint)
    registry = tmp_path / "models.toml"
    registry.write_text(
        f'[[models]]\nname = "Broken"\npath = "{checkpoint.name}"\n',
        encoding="utf-8",
    )
    client = TestClient(create_app(registry))

    response = client.post("/api/models/broken/load")

    assert response.status_code == 422
    assert "embedded experiment configuration" in response.json()["detail"]


def test_api_rejects_invalid_context_and_overlap(tmp_path: Path) -> None:
    """ Verify rating increments, duplicate IDs, and query overlap are rejected.

    Args:
        tmp_path: Temporary test directory.
    """
    client = _client(tmp_path)
    client.post("/api/models/first-cut/load")

    invalid_rating = client.post(
        "/api/predictions/manual",
        json={"context": [{"movie_id": "m1", "rating": 4.2}], "query_movie_ids": ["m2"]},
    )
    overlap = client.post(
        "/api/predictions/manual",
        json={"context": _context(), "query_movie_ids": ["m1"]},
    )

    assert invalid_rating.status_code == 422
    assert overlap.status_code == 422
