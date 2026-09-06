""" FastAPI application for the local LettMePick testing workbench."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Self

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from src.ui.context_store import ContextStore
from src.ui.runtime import ModelRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "models.toml"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
DEFAULT_CONTEXT_DIR = PROJECT_ROOT / "saved_contexts"
MAX_CONTEXT_SIZE = 512


class ContextRating(BaseModel):
    """ Pair one context movie with a half-star rating."""

    movie_id: str = Field(min_length=1)
    rating: float = Field(ge=0.5, le=5.0)

    @field_validator("rating")
    @classmethod
    def validate_half_star(cls, value: float) -> float:
        """ Require ratings to use half-star increments.

        Args:
            value: Submitted five-star rating.

        Returns:
            Validated rating.
        """
        if abs(value * 2 - round(value * 2)) > 1e-6:
            raise ValueError("rating must use 0.5-star increments")
        return value


class ManualPredictionRequest(BaseModel):
    """ Describe an explicit context and manual query set."""

    context: list[ContextRating] = Field(min_length=1, max_length=MAX_CONTEXT_SIZE)
    query_movie_ids: list[str] = Field(min_length=1, max_length=512)


class AutoRunRequest(BaseModel):
    """ Describe context and sampling size for an automatic run."""

    context: list[ContextRating] = Field(min_length=1, max_length=MAX_CONTEXT_SIZE)
    batch_size: int = Field(default=64, ge=1, le=512)
    min_year: int | None = None
    max_year: int | None = None
    genres: list[str] = Field(default_factory=list, max_length=100)
    min_rating_count: int = Field(default=0, ge=0)

    @field_validator("genres")
    @classmethod
    def normalize_genres(cls, values: list[str]) -> list[str]:
        """ Normalize and deduplicate selected genres.

        Args:
            values: Submitted genre labels.

        Returns:
            Normalized unique labels.
        """
        normalized = [value.strip().casefold() for value in values]
        if any(not value for value in normalized):
            raise ValueError("genres cannot contain empty values")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_year_range(self) -> Self:
        """ Require an ordered inclusive year range.

        Returns:
            Validated request.
        """
        if (
            self.min_year is not None
            and self.max_year is not None
            and self.min_year > self.max_year
        ):
            raise ValueError("min_year cannot be greater than max_year")
        return self


class SaveContextRequest(BaseModel):
    """ Describe one named context to persist on the local server."""

    name: str = Field(min_length=1, max_length=60)
    model_id: str = Field(min_length=1)
    context: list[ContextRating] = Field(min_length=1, max_length=MAX_CONTEXT_SIZE)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """ Strip and require a visible context name.

        Args:
            value: Submitted context name.

        Returns:
            Trimmed context name.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("name cannot contain only whitespace")
        return stripped


def _context_values(context: list[ContextRating]) -> list[tuple[str, float]]:
    """ Convert validated API context objects into runtime tuples.

    Args:
        context: Validated context payload.

    Returns:
        Movie IDs paired with ratings.
    """
    values = [(item.movie_id, item.rating) for item in context]
    ids = [movie_id for movie_id, _ in values]
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=422, detail="Context movies cannot contain duplicates.")
    return values


def create_app(
    registry_path: Path = DEFAULT_REGISTRY,
    context_dir: Path = DEFAULT_CONTEXT_DIR,
) -> FastAPI:
    """ Create an application around one local model runtime.

    Args:
        registry_path: Trusted model registry path.
        context_dir: Directory used for persistent local contexts.

    Returns:
        Configured FastAPI application.
    """
    runtime = ModelRuntime(registry_path)
    context_store = ContextStore(context_dir)
    application = FastAPI(
        title="LettMePick Screening Room",
        version="0.1.0",
        description="Local cinematic workbench for testing recommendation checkpoints.",
    )
    application.state.runtime = runtime
    application.state.context_store = context_store
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/models")
    def models() -> dict[str, object]:
        """ List registered model metadata.

        Returns:
            Registry summaries and active state.
        """
        return {"models": runtime.list_models()}

    @application.get("/api/models/{model_id}/config")
    def inspect_config(model_id: str) -> dict[str, str]:
        """ Return the optional registry override configuration as text.

        Args:
            model_id: Registry identifier.

        Returns:
            Registry override configuration text.
        """
        try:
            return {"content": runtime.inspect_config(model_id)}
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (FileNotFoundError, UnicodeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/api/models/{model_id}/load")
    def load_model(model_id: str) -> dict[str, object]:
        """ Lazily load one trusted checkpoint and its catalog.

        Args:
            model_id: Registry identifier.

        Returns:
            Loaded model metadata.
        """
        try:
            return runtime.load_model(model_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get("/api/contexts")
    def contexts() -> dict[str, object]:
        """ List contexts persisted on the local server.

        Returns:
            Saved contexts ordered by most recent update.
        """
        try:
            return {"contexts": context_store.list_contexts()}
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @application.post("/api/contexts")
    def save_context(payload: SaveContextRequest) -> dict[str, object]:
        """ Save or update one context using canonical catalog metadata.

        Args:
            payload: Context name, model, movie IDs, and ratings.

        Returns:
            Persisted context record.
        """
        try:
            context = _context_values(payload.context)
            model_name, movies = runtime.context_record(payload.model_id, context)
            return context_store.save_context(
                payload.name, payload.model_id, model_name, movies
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except OSError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @application.delete("/api/contexts/{context_id}", status_code=204)
    def delete_context(context_id: str) -> None:
        """ Delete one persisted context.

        Args:
            context_id: Saved context identifier.
        """
        try:
            context_store.delete_context(context_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @application.get("/api/movies")
    def movies(
        query: Annotated[str, Query(min_length=1, max_length=200)],
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
    ) -> dict[str, object]:
        """ Search the active model's aligned movie catalog.

        Args:
            query: Partial or complete movie title.
            limit: Maximum result count.

        Returns:
            Matching movie records.
        """
        try:
            return {"movies": runtime.search_movies(query, limit)}
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.get("/api/movie-facets")
    def movie_facets() -> dict[str, object]:
        """ Return filter values available for the active movie catalog.

        Returns:
            Inclusive year bounds and sorted genre labels.
        """
        try:
            return runtime.movie_facets()
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.post("/api/predictions/manual")
    def manual_predictions(payload: ManualPredictionRequest) -> dict[str, object]:
        """ Score and rank explicitly selected query movies.

        Args:
            payload: Rated context and explicit query IDs.

        Returns:
            Ranked predictions.
        """
        context = _context_values(payload.context)
        context_ids = {movie_id for movie_id, _ in context}
        overlap = context_ids.intersection(payload.query_movie_ids)
        if overlap:
            raise HTTPException(
                status_code=422,
                detail=f"Context and query movies must be disjoint: {sorted(overlap)[:5]}",
            )
        try:
            return {"results": runtime.manual_predict(context, payload.query_movie_ids)}
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/api/auto-runs")
    def start_auto_run(payload: AutoRunRequest) -> dict[str, object]:
        """ Start automatic sampling and return its first accumulated ranking.

        Args:
            payload: Rated context and batch size.

        Returns:
            Initial automatic-run state.
        """
        try:
            return runtime.start_auto_run(
                _context_values(payload.context),
                payload.batch_size,
                min_year=payload.min_year,
                max_year=payload.max_year,
                genres=payload.genres,
                min_rating_count=payload.min_rating_count,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/api/auto-runs/{run_id}/next")
    def next_auto_batch(run_id: str) -> dict[str, object]:
        """ Score the next unseen random batch.

        Args:
            run_id: Active run identifier.

        Returns:
            Updated automatic-run state.
        """
        try:
            return runtime.next_auto_batch(run_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.delete("/api/auto-runs/{run_id}", status_code=204)
    def stop_auto_run(run_id: str) -> None:
        """ Stop automatic sampling and release cached state.

        Args:
            run_id: Active run identifier.
        """
        try:
            runtime.stop_auto_run(run_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    if FRONTEND_DIST.is_dir():
        assets = FRONTEND_DIST / "assets"
        if assets.is_dir():
            application.mount("/assets", StaticFiles(directory=assets), name="assets")

        @application.get("/{path:path}", include_in_schema=False)
        def frontend(path: str) -> FileResponse:
            """ Serve the built single-page application with history fallback.

            Args:
                path: Requested frontend-relative path.

            Returns:
                Static asset or application entry point.
            """
            requested = FRONTEND_DIST / path
            if path and requested.is_file() and FRONTEND_DIST in requested.resolve().parents:
                return FileResponse(requested)
            return FileResponse(FRONTEND_DIST / "index.html")

    return application


app = create_app(
    Path(os.environ.get("LETTMEPICK_MODEL_REGISTRY", DEFAULT_REGISTRY)),
    Path(os.environ.get("LETTMEPICK_CONTEXT_DIR", DEFAULT_CONTEXT_DIR)),
)
