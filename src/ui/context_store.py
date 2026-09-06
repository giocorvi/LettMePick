""" Filesystem persistence for locally saved model-testing contexts."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


MAX_CONTEXT_SIZE = 512


class StoredContextMovie(BaseModel):
    """ Store one canonical movie and its user rating."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    year: int
    genres: list[str]
    rating: float = Field(ge=0.5, le=5.0)

    @field_validator("rating")
    @classmethod
    def validate_half_star(cls, value: float) -> float:
        """ Require ratings to use half-star increments.

        Args:
            value: Submitted rating.

        Returns:
            Validated rating.
        """
        if abs(value * 2 - round(value * 2)) > 1e-6:
            raise ValueError("rating must use 0.5-star increments")
        return value


class StoredContext(BaseModel):
    """ Store one named context associated with a registered model."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=60)
    model_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    movies: list[StoredContextMovie] = Field(min_length=1, max_length=MAX_CONTEXT_SIZE)
    updated_at: datetime


class ContextStore:
    """ Atomically persist a bounded collection of contexts in one JSON file."""

    def __init__(self, directory: Path):
        """ Configure a context store without creating it eagerly.

        Args:
            directory: Directory receiving ``contexts.json``.
        """
        self.directory = directory.resolve()
        self.path = self.directory / "contexts.json"
        self._lock = threading.RLock()

    def list_contexts(self) -> list[dict[str, object]]:
        """ Return saved contexts ordered by most recent update.

        Returns:
            Serializable saved-context records.
        """
        with self._lock:
            contexts = sorted(self._read(), key=lambda item: item.updated_at, reverse=True)
            return [context.model_dump(mode="json") for context in contexts]

    def save_context(
        self,
        name: str,
        model_id: str,
        model_name: str,
        movies: list[dict[str, object]],
    ) -> dict[str, object]:
        """ Create or replace a context with the same model and name.

        Args:
            name: User-facing context name.
            model_id: Registered model identifier.
            model_name: Registered model display name.
            movies: Canonical movie metadata and ratings.

        Returns:
            Saved context record.
        """
        normalized_name = name.strip()
        with self._lock:
            contexts = self._read()
            existing = next(
                (
                    item
                    for item in contexts
                    if item.model_id == model_id
                    and item.name.casefold() == normalized_name.casefold()
                ),
                None,
            )
            if existing is None and len(contexts) >= 40:
                raise ValueError("At most 40 contexts can be saved.")
            saved = StoredContext(
                id=existing.id if existing else uuid.uuid4().hex,
                name=normalized_name,
                model_id=model_id,
                model_name=model_name,
                movies=[StoredContextMovie.model_validate(movie) for movie in movies],
                updated_at=datetime.now(UTC),
            )
            contexts = [item for item in contexts if item.id != saved.id]
            contexts.append(saved)
            self._write(contexts)
            return saved.model_dump(mode="json")

    def delete_context(self, context_id: str) -> None:
        """ Delete one context by identifier.

        Args:
            context_id: Saved context identifier.
        """
        with self._lock:
            contexts = self._read()
            retained = [item for item in contexts if item.id != context_id]
            if len(retained) == len(contexts):
                raise LookupError(f"Saved context does not exist: {context_id}")
            self._write(retained)

    def _read(self) -> list[StoredContext]:
        """ Read and validate the on-disk document.

        Returns:
            Validated stored contexts.
        """
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("version") != 1:
                raise ValueError("unsupported saved-context file version")
            contexts = raw.get("contexts")
            if not isinstance(contexts, list):
                raise ValueError("saved-context file must contain a contexts list")
            return [StoredContext.model_validate(item) for item in contexts]
        except (json.JSONDecodeError, OSError, ValueError) as error:
            raise ValueError(f"Could not read saved contexts from {self.path}: {error}") from error

    def _write(self, contexts: list[StoredContext]) -> None:
        """ Atomically write the current context collection.

        Args:
            contexts: Validated contexts to persist.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        document = {
            "version": 1,
            "contexts": [context.model_dump(mode="json") for context in contexts],
        }
        try:
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
