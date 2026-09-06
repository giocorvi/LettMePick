""" Trusted model registry and in-memory inference runtime."""

from __future__ import annotations

import gc
import random
import re
import threading
import tomllib
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.data.data import (
    MovieCatalogEntry,
    POPULARITY_LEVELS,
    build_movie_catalog,
    filter_movie_catalog,
    prepare_dataset,
)
from src.data.prehash import collate_prehashed_bank
from src.models.lett_me_pick import LettMePick
from src.utils.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
    load_config,
    validate_config,
)
from src.utils.train import build_model, resolve_device


@dataclass(frozen=True)
class RegisteredModel:
    """ Describe one trusted checkpoint exposed by the registry."""

    id: str
    name: str
    path: Path
    config: Path | None
    description: str | None


@dataclass
class LoadedModel:
    """ Hold the active model and its aligned searchable data."""

    registered: RegisteredModel
    model: LettMePick
    bank: dict[str, Tensor]
    catalog: list[MovieCatalogEntry]
    catalog_by_id: dict[str, MovieCatalogEntry]
    device: torch.device
    checkpoint_config: ExperimentConfig


@dataclass
class AutoRun:
    """ Retain cached context and accumulated scores for one auto run."""

    id: str
    encoded_context: Tensor
    remaining_indices: list[int]
    batch_size: int
    results: dict[int, float]
    batches: int = 0


def _slug(value: str) -> str:
    """ Convert a model name into a stable URL identifier.

    Args:
        value: Human-readable model name.

    Returns:
        A lowercase URL-safe identifier.
    """
    result = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return result or "model"


def load_registry(path: Path) -> list[RegisteredModel]:
    """ Load and validate trusted model metadata from TOML.

    Args:
        path: Registry file path.

    Returns:
        Validated registered models.
    """
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model registry does not exist: {resolved}")
    with resolved.open("rb") as file:
        raw = tomllib.load(file)
    entries = raw.get("models", [])
    if not isinstance(entries, list):
        raise ValueError("Registry 'models' must be an array of tables.")

    allowed = {"name", "path", "config", "description"}
    seen_ids: set[str] = set()
    models: list[RegisteredModel] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Registry model at index {index} must be a table.")
        unexpected = set(entry) - allowed
        if unexpected:
            raise ValueError(f"Registry model {index} has unsupported fields: {sorted(unexpected)}")
        if not entry.get("name") or not entry.get("path"):
            raise ValueError(f"Registry model {index} requires non-empty name and path.")
        model_id = _slug(str(entry["name"]))
        if model_id in seen_ids:
            raise ValueError(f"Registry model names must produce unique IDs: {model_id}")
        seen_ids.add(model_id)

        def registry_path(value: str | None) -> Path | None:
            """ Resolve one optional path relative to the registry.

            Args:
                value: Configured path text.

            Returns:
                Resolved path or ``None``.
            """
            if value is None:
                return None
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (resolved.parent / candidate).resolve()

        models.append(
            RegisteredModel(
                id=model_id,
                name=str(entry["name"]),
                path=registry_path(str(entry["path"])),  # type: ignore[arg-type]
                config=registry_path(str(entry["config"])) if entry.get("config") else None,
                description=str(entry["description"]) if entry.get("description") else None,
            )
        )
    return models


def _embedded_config(raw: Any) -> ExperimentConfig:
    """ Reconstruct and validate an experiment config embedded in a checkpoint.

    Args:
        raw: Untrusted-in-shape checkpoint configuration value.

    Returns:
        Validated experiment configuration.
    """
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint is missing its embedded experiment configuration.")
    required = {"data", "model", "training", "runtime"}
    if not required.issubset(raw):
        raise ValueError("Checkpoint configuration must contain data, model, training, and runtime sections.")
    try:
        data_values = dict(raw["data"])
        for key in ("ratings_path", "movies_path", "cache_path"):
            if data_values.get(key) is not None:
                data_values[key] = Path(data_values[key]).expanduser().resolve()
        runtime_values = dict(raw["runtime"])
        for key in ("log_path", "checkpoint_dir", "resume_path"):
            if runtime_values.get(key) is not None:
                runtime_values[key] = Path(runtime_values[key]).expanduser().resolve()
        config = ExperimentConfig(
            data=DataConfig(**data_values),
            model=ModelConfig(**dict(raw["model"])),
            training=TrainingConfig(**dict(raw["training"])),
            runtime=RuntimeConfig(**runtime_values),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Checkpoint contains an invalid experiment configuration: {error}") from error
    validate_config(config)
    return config


class ModelRuntime:
    """ Coordinate one lazily loaded model and one local inference run."""

    def __init__(self, registry_path: Path):
        """ Load registry metadata without loading any checkpoints.

        Args:
            registry_path: Trusted registry TOML path.
        """
        self.registry_path = registry_path.resolve()
        self.models = load_registry(self.registry_path)
        self.models_by_id = {model.id: model for model in self.models}
        self.active: LoadedModel | None = None
        self.auto_run: AutoRun | None = None
        self.loading_model_id: str | None = None
        self.loading_stage: str | None = None
        self._lock = threading.RLock()
        self._random = random.SystemRandom()

    def list_models(self) -> list[dict[str, Any]]:
        """ Return serializable registry metadata and active state.

        Returns:
            Model summaries for the UI.
        """
        return [
            {
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "has_config": model.config is not None,
                "active": self.active is not None and self.active.registered.id == model.id,
                "loading": self.loading_model_id == model.id,
                "loading_stage": (
                    self.loading_stage if self.loading_model_id == model.id else None
                ),
            }
            for model in self.models
        ]

    def inspect_config(self, model_id: str) -> str:
        """ Read a registry configuration override for display.

        Args:
            model_id: Registered model identifier.

        Returns:
            UTF-8 configuration text.
        """
        registered = self._registered(model_id)
        if registered.config is None:
            raise LookupError("This model has no registry config override.")
        if not registered.config.is_file():
            raise FileNotFoundError(
                f"Registry config override does not exist: {registered.config}"
            )
        return registered.config.read_text(encoding="utf-8")

    def load_model(self, model_id: str) -> dict[str, Any]:
        """ Load a registered checkpoint or a module stored directly in one.

        A registry config takes precedence over checkpoint configuration. Without an
        override, checkpoint dictionaries use their embedded configuration and direct
        modules must expose an ``experiment_config`` attribute.

        Args:
            model_id: Registered model identifier.

        Returns:
            Active model metadata.
        """
        with self._lock:
            registered = self._registered(model_id)
            self.loading_model_id = model_id
            self.loading_stage = "Reading configuration"
            try:
                if not registered.path.is_file():
                    raise FileNotFoundError(f"Checkpoint does not exist: {registered.path}")
                override_config = (
                    load_config(registered.config) if registered.config is not None else None
                )
                self.loading_stage = "Loading checkpoint"
                saved = torch.load(registered.path, map_location="cpu", weights_only=False)
                if isinstance(saved, LettMePick):
                    checkpoint_config = override_config or getattr(saved, "experiment_config", None)
                    if not isinstance(checkpoint_config, ExperimentConfig):
                        raise ValueError(
                            "A saved LettMePick module must define an experiment_config "
                            "attribute containing its originating ExperimentConfig."
                        )
                    validate_config(checkpoint_config)
                    return self._activate_model(registered, saved, checkpoint_config)
                if isinstance(saved, torch.nn.Module):
                    raise TypeError(
                        "A saved module must be a LettMePick instance so cached-context "
                        "inference remains available."
                    )
                if not isinstance(saved, dict):
                    raise ValueError(
                        "Checkpoint must contain a dictionary payload or LettMePick module."
                    )
                checkpoint_config = override_config or _embedded_config(saved.get("config"))
                model = build_model(checkpoint_config)
                state = saved.get("best_model_state_dict") or saved.get("model_state_dict")
                if not isinstance(state, dict):
                    raise ValueError(
                        "Checkpoint has neither best_model_state_dict nor model_state_dict."
                    )
                model.load_state_dict(state)
                return self._activate_model(registered, model, checkpoint_config)
            finally:
                self.loading_model_id = None
                self.loading_stage = None

    def _activate_model(
        self,
        registered: RegisteredModel,
        model: LettMePick,
        config: ExperimentConfig,
    ) -> dict[str, Any]:
        """ Prepare data and activate one already-constructed model.

        Args:
            registered: Display metadata for the active model.
            model: Model with compatible cached-context inference methods.
            config: Experiment configuration that defines movie data and hashing.

        Returns:
            Active model metadata.
        """
        device = resolve_device(config.runtime.device)
        movies_path = config.data.movies_path
        if not movies_path.is_file():
            raise FileNotFoundError(f"Configured movie dataset does not exist: {movies_path}")
        cached = self._load_movie_cache(config)
        if cached is None:
            self.loading_stage = "Reading movie metadata"
            movies = torch.load(movies_path, map_location="cpu", weights_only=False)
            if not isinstance(movies, dict):
                raise TypeError("Configured movie dataset must contain a dictionary.")
            self.loading_stage = "Preparing movie bank (first load)"
            bank = prepare_dataset(
                {},
                movies,
                num_id_buckets=config.data.num_id_buckets,
                num_actor_buckets=config.data.num_actor_buckets,
                num_genre_buckets=config.data.num_genre_buckets,
                num_director_buckets=config.data.num_director_buckets,
            )
            # Build display objects after hashing to reduce peak memory substantially.
            catalog = build_movie_catalog(movies)
            del movies
            gc.collect()
            self.loading_stage = "Saving reusable movie cache"
            self._save_movie_cache(config, catalog, bank)
        else:
            cache_version, catalog, bank = cached
            if cache_version == 2:
                self.loading_stage = "Adding popularity metadata (one-time)"
                expected_catalog_length = len(catalog)
                del catalog
                gc.collect()
                movies = torch.load(movies_path, map_location="cpu", weights_only=False)
                if not isinstance(movies, dict):
                    raise TypeError("Configured movie dataset must contain a dictionary.")
                upgraded_catalog = build_movie_catalog(movies)
                del movies
                gc.collect()
                if (
                    len(upgraded_catalog) != expected_catalog_length
                    or len(upgraded_catalog) != int(bank["id_idx"].shape[0])
                    or any(
                        entry.bank_index != index
                        for index, entry in enumerate(upgraded_catalog)
                    )
                ):
                    raise ValueError(
                        "Version-2 movie cache is not aligned with the current raw catalog."
                    )
                catalog = upgraded_catalog
                self.loading_stage = "Saving upgraded movie cache"
                self._save_movie_cache(config, catalog, bank)
        self.loading_stage = f"Moving model to {device}"
        model.to(device).eval()
        self.auto_run = None
        self.active = LoadedModel(
            registered=registered,
            model=model,
            bank=bank,
            catalog=catalog,
            catalog_by_id={entry.id: entry for entry in catalog},
            device=device,
            checkpoint_config=config,
        )
        return {
            "id": registered.id,
            "name": registered.name,
            "description": registered.description,
            "catalog_size": len(catalog),
            "device": str(device),
            "architecture": asdict(config.model),
        }

    def _movie_cache_path(self, config: ExperimentConfig) -> Path:
        """ Return the derived UI movie-cache path.

        Args:
            config: Effective experiment configuration.

        Returns:
            Cache path beside the configured movie metadata file.
        """
        movies_path = config.data.movies_path
        return movies_path.with_name(f".{movies_path.name}.lettmepick-ui-cache.pt")

    def _movie_cache_key(self, config: ExperimentConfig) -> dict[str, Any]:
        """ Describe inputs that invalidate prepared UI movie data.

        Args:
            config: Effective experiment configuration.

        Returns:
            Serializable source and hashing metadata.
        """
        stat = config.data.movies_path.stat()
        return {
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "buckets": {
                "id": config.data.num_id_buckets,
                "actor": config.data.num_actor_buckets,
                "genre": config.data.num_genre_buckets,
                "director": config.data.num_director_buckets,
            },
        }

    def _load_movie_cache(
        self, config: ExperimentConfig
    ) -> tuple[int, list[MovieCatalogEntry], dict[str, Tensor]] | None:
        """ Load prepared movie data when its source signature still matches.

        Args:
            config: Effective experiment configuration.

        Returns:
            Cache version, catalog, and tensor bank, or ``None`` when rebuilding is required.
        """
        path = self._movie_cache_path(config)
        if not path.is_file():
            return None
        self.loading_stage = "Loading prepared movie cache"
        try:
            cached = torch.load(path, map_location="cpu", weights_only=False)
        except (EOFError, OSError, RuntimeError):
            return None
        if not isinstance(cached, dict) or cached.get("version") not in (2, 3):
            return None
        if cached.get("key") != self._movie_cache_key(config):
            return None
        raw_catalog = cached.get("catalog")
        bank = cached.get("bank")
        if not isinstance(raw_catalog, list) or not isinstance(bank, dict):
            return None
        if not all(isinstance(entry, MovieCatalogEntry) for entry in raw_catalog):
            return None
        catalog = raw_catalog
        required = {
            "id_idx",
            "year",
            "actors_idx",
            "actors_mask",
            "genres_idx",
            "genres_mask",
            "directors_idx",
            "directors_mask",
        }
        if not required.issubset(bank) or not all(
            isinstance(value, Tensor) for value in bank.values()
        ):
            return None
        if len(catalog) != int(bank["id_idx"].shape[0]) or any(
            entry.bank_index != index for index, entry in enumerate(catalog)
        ):
            return None
        return int(cached["version"]), catalog, bank

    def _save_movie_cache(
        self,
        config: ExperimentConfig,
        catalog: list[MovieCatalogEntry],
        bank: dict[str, Tensor],
    ) -> None:
        """ Best-effort persist prepared UI movie data for later launches.

        Args:
            config: Effective experiment configuration.
            catalog: Searchable movie entries aligned with the bank.
            bank: Prepared movie feature tensors.
        """
        path = self._movie_cache_path(config)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(
                {
                    "version": 3,
                    "key": self._movie_cache_key(config),
                    "catalog": catalog,
                    "bank": bank,
                },
                temporary,
            )
            temporary.replace(path)
        except (OSError, RuntimeError):
            temporary.unlink(missing_ok=True)

    def search_movies(self, query: str, limit: int) -> list[dict[str, Any]]:
        """ Search active catalog titles with prefix matches first.

        Args:
            query: Partial or complete title.
            limit: Maximum returned matches.

        Returns:
            Matching movie summaries.
        """
        active = self._active()
        normalized = query.strip().casefold()
        if not normalized:
            return []
        matches = [entry for entry in active.catalog if normalized in entry.title.casefold()]
        matches.sort(
            key=lambda entry: (
                not entry.title.casefold().startswith(normalized),
                entry.title.casefold(),
                entry.year,
            )
        )
        return [self._movie(entry) for entry in matches[:limit]]

    def movie_facets(self) -> dict[str, object]:
        """ Return backend-owned facets for auto candidate filtering.

        Returns:
            Model-specific movie-filter metadata.
        """
        catalog = self._active().catalog
        return {
            "min_year": min(entry.year for entry in catalog),
            "max_year": max(entry.year for entry in catalog),
            "genres": sorted({genre for entry in catalog for genre in entry.genres}),
            "popularity_levels": [dict(level) for level in POPULARITY_LEVELS],
        }

    def context_record(
        self, model_id: str, context: list[tuple[str, float]]
    ) -> tuple[str, list[dict[str, object]]]:
        """ Resolve a rated context to canonical catalog metadata for persistence.

        Args:
            model_id: Model the saved context belongs to.
            context: Movie IDs paired with five-star ratings.

        Returns:
            Registered model name and canonical rated movie records.
        """
        registered = self._registered(model_id)
        active = self._active()
        if active.registered.id != model_id:
            raise RuntimeError(f"Load {registered.name} before saving a context for it.")
        movies: list[dict[str, object]] = []
        for movie_id, rating in context:
            entry = active.catalog_by_id.get(movie_id)
            if entry is None:
                raise ValueError(f"Unknown movie ID: {movie_id}")
            movies.append({**self._movie(entry), "rating": rating})
        return registered.name, movies

    def manual_predict(
        self, context: list[tuple[str, float]], query_ids: list[str]
    ) -> list[dict[str, Any]]:
        """ Score explicit query movies against one rated context.

        Args:
            context: Movie IDs paired with half-star ratings.
            query_ids: Explicit query movie IDs.

        Returns:
            Ranked prediction records.
        """
        with self._lock, torch.inference_mode():
            active = self._active()
            context_indices, scores = self._context_tensors(active, context)
            query_indices = self._indices(active, query_ids)
            context_batch = collate_prehashed_bank([context_indices], active.bank, active.device)
            query_batch = collate_prehashed_bank([query_indices], active.bank, active.device)
            encoded = active.model.encode_context(context_batch, scores)
            predictions = active.model.score_encoded_context(encoded, query_batch)[0]
            return self._rank(active, dict(zip(query_indices, predictions.tolist())))

    def start_auto_run(
        self,
        context: list[tuple[str, float]],
        batch_size: int,
        min_year: int | None = None,
        max_year: int | None = None,
        genres: list[str] | None = None,
        min_rating_count: int = 0,
    ) -> dict[str, Any]:
        """ Cache context and score the first unseen random batch.

        Args:
            context: Movie IDs paired with half-star ratings.
            batch_size: Maximum candidates in each batch.
            min_year: Optional inclusive earliest candidate year.
            max_year: Optional inclusive latest candidate year.
            genres: Optional genres matched with any-overlap semantics.
            min_rating_count: Inclusive minimum Letterboxd rating count.

        Returns:
            Auto-run state after its first batch.
        """
        with self._lock, torch.inference_mode():
            active = self._active()
            context_indices, scores = self._context_tensors(active, context)
            available_genres = {genre for entry in active.catalog for genre in entry.genres}
            selected_genres = {genre.casefold() for genre in genres or []}
            unknown_genres = selected_genres - available_genres
            if unknown_genres:
                raise ValueError(f"Unknown genres: {sorted(unknown_genres)}")
            eligible = filter_movie_catalog(
                active.catalog,
                min_year=min_year,
                max_year=max_year,
                genres=list(selected_genres),
                min_rating_count=min_rating_count,
            )
            excluded = set(context_indices)
            remaining = [
                entry.bank_index for entry in eligible if entry.bank_index not in excluded
            ]
            if not remaining:
                raise ValueError("No candidate movies match the selected auto filters.")
            context_batch = collate_prehashed_bank([context_indices], active.bank, active.device)
            encoded = active.model.encode_context(context_batch, scores)
            run = AutoRun(
                id=uuid.uuid4().hex,
                encoded_context=encoded,
                remaining_indices=remaining,
                batch_size=batch_size,
                results={},
            )
            self.auto_run = run
            return self._score_next(active, run)

    def next_auto_batch(self, run_id: str) -> dict[str, Any]:
        """ Score the next batch for the active auto run.

        Args:
            run_id: Active run identifier.

        Returns:
            Updated accumulated ranking.
        """
        with self._lock, torch.inference_mode():
            run = self._run(run_id)
            return self._score_next(self._active(), run)

    def stop_auto_run(self, run_id: str) -> None:
        """ Stop an auto run and release its cached tensor.

        Args:
            run_id: Active run identifier.
        """
        with self._lock:
            self._run(run_id)
            self.auto_run = None

    def _score_next(self, active: LoadedModel, run: AutoRun) -> dict[str, Any]:
        """ Sample, score, and serialize one auto batch.

        Args:
            active: Currently loaded model resources.
            run: Active automatic inference state.

        Returns:
            Updated serialized run state.
        """
        if run.remaining_indices:
            count = min(run.batch_size, len(run.remaining_indices))
            indices = self._random.sample(run.remaining_indices, count)
            selected = set(indices)
            run.remaining_indices = [index for index in run.remaining_indices if index not in selected]
            query = collate_prehashed_bank([indices], active.bank, active.device)
            predictions = active.model.score_encoded_context(run.encoded_context, query)[0]
            run.results.update(zip(indices, predictions.tolist()))
            run.batches += 1
        return {
            "run_id": run.id,
            "batches": run.batches,
            "sampled": len(run.results),
            "remaining": len(run.remaining_indices),
            "complete": not run.remaining_indices,
            "results": self._rank(active, run.results),
        }

    def _context_tensors(
        self, active: LoadedModel, context: list[tuple[str, float]]
    ) -> tuple[list[int], Tensor]:
        """ Validate context IDs and create normalized scores.

        Args:
            active: Currently loaded model resources.
            context: Movie IDs paired with five-star ratings.

        Returns:
            Bank rows and normalized score tensor.
        """
        ids = [movie_id for movie_id, _ in context]
        indices = self._indices(active, ids)
        values = torch.tensor(
            [[rating / 5.0 for _, rating in context]],
            dtype=torch.float32,
            device=active.device,
        )
        return indices, values

    def _indices(self, active: LoadedModel, movie_ids: list[str]) -> list[int]:
        """ Resolve unique catalog IDs into bank rows.

        Args:
            active: Currently loaded model resources.
            movie_ids: Unique movie identifiers.

        Returns:
            Corresponding tensor-bank rows.
        """
        if len(movie_ids) != len(set(movie_ids)):
            raise ValueError("Movie selections cannot contain duplicates.")
        missing = [movie_id for movie_id in movie_ids if movie_id not in active.catalog_by_id]
        if missing:
            raise ValueError(f"Unknown movie IDs: {missing[:5]}")
        return [active.catalog_by_id[movie_id].bank_index for movie_id in movie_ids]

    def _rank(self, active: LoadedModel, scores: dict[int, float]) -> list[dict[str, Any]]:
        """ Serialize bank-index scores in decreasing order.

        Args:
            active: Currently loaded model resources.
            scores: Predictions keyed by tensor-bank row.

        Returns:
            Ranked display records.
        """
        ordered = sorted(scores.items(), key=lambda item: (-item[1], active.catalog[item[0]].title.casefold()))
        return [
            {**self._movie(active.catalog[index]), "score": score, "score_five": score * 5.0}
            for index, score in ordered
        ]

    @staticmethod
    def _movie(entry: MovieCatalogEntry) -> dict[str, Any]:
        """ Serialize one catalog entry.

        Args:
            entry: Catalog metadata to expose.

        Returns:
            JSON-compatible movie metadata.
        """
        return {
            "id": entry.id,
            "title": entry.title,
            "year": entry.year,
            "genres": entry.genres,
        }

    def _registered(self, model_id: str) -> RegisteredModel:
        """ Resolve a model ID or raise a lookup error.

        Args:
            model_id: Registry identifier.

        Returns:
            Matching registry entry.
        """
        try:
            return self.models_by_id[model_id]
        except KeyError as error:
            raise LookupError(f"Unknown model: {model_id}") from error

    def _active(self) -> LoadedModel:
        """ Return the active model or require one to be loaded.

        Returns:
            Currently loaded model resources.
        """
        if self.active is None:
            raise RuntimeError("Load a model before using the movie catalog or inference.")
        return self.active

    def _run(self, run_id: str) -> AutoRun:
        """ Resolve the sole active auto run.

        Args:
            run_id: Expected active run identifier.

        Returns:
            Active run state.
        """
        if self.auto_run is None or self.auto_run.id != run_id:
            raise LookupError("Auto run is no longer active.")
        return self.auto_run
