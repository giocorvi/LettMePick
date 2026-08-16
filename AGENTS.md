# Repository Guide for Agents

## Project Overview

LettMePick is an experimental PyTorch movie-recommendation project. The supported model predicts ratings for query movies from a user's previously rated context movies. The repository is being rebuilt around scalable data preparation, repeatable model training and deployment, and a future UI as the primary interaction surface.

The current repository contains the data and model core only. There is no supported training CLI, inference API, deployment workflow, or UI yet. Do not assume those interfaces exist.

## Supported Architecture

The supported flow is:

1. Raw movie metadata and user ratings enter `src/data/data.py`.
2. `prepare_dataset` filters movies, normalizes ratings to `[0, 1]`, remaps movie IDs to bank rows, and creates a hashed movie bank.
3. `src/data/prehash.py` hashes movie IDs, actors, genres, and directors deterministically and packs them into fixed-width tensors with masks.
4. `get_train_batch` samples rated context movies and held-out query movies, then collates their hashed features.
5. `src/models/movie_encoder.py` builds feature tokens and produces one normalized embedding per movie.
6. `src/models/lett_me_pick.py` adds learned score embeddings to rated context movies, then uses self-attention over that context and cross-attention from query movies to predict ratings.

The old dense-embedding pipeline was deliberately removed. Do not reintroduce `MovieEmbedder`, `SimpleMovieEncoder`, `data_v2`, `prehash_v2`, or the old experimental model variants unless explicitly requested.

## Important Data Contracts

- `prepare_dataset` mutates the supplied user-ratings mapping in place.
- Raw ratings use a ten-point scale; prepared ratings are normalized to `[0, 1]`.
- Movies without `year_released` and ratings referencing unavailable movies are omitted.
- A movie bank contains `id_idx`, `year`, and actor/genre/director index and mask tensors.
- Collated movie tensors flatten the batch and movie-sequence dimensions. `MovieEncoder` restores them from explicit `batch_size` and `sequence_size` values.
- Feature hashing uses SHA-256 and must remain deterministic across processes and platforms.

## Repository Map

- `src/data/data.py`: Raw metadata preparation, rating remapping, user splitting, and batch sampling.
- `src/data/prehash.py`: Deterministic hashing, tensor-bank construction, and indexed collation.
- `src/losses/`: Standalone rating-regression and pairwise-ranking objectives.
- `src/models/movie_encoder.py`: Metadata tokenization, movie attention, and final movie projection.
- `src/models/self_gpt.py`: Self-attention and masked self-attention blocks.
- `src/models/cross_gpt.py`: Cross-attention blocks.
- `src/models/lett_me_pick.py`: End-to-end recommendation model, losses, and cached inference.
- `src/inspect_*.py`: Standalone tools for inspecting external movie and ratings datasets.
- `tests/`: Unit and integration coverage for the supported data and model path.

## Environment and Commands

Python 3.12 or newer is required. Dependencies are managed with `uv`.

```bash
uv sync
uv run pytest -q
git diff --check
```

Run targeted tests while iterating, then run the full suite before handing work back:

```bash
uv run pytest -q tests/test_data.py
uv run pytest -q tests/test_prehash.py
uv run pytest -q tests/test_losses.py
uv run pytest -q tests/test_model.py
```

No formatter, linter, type checker, CI pipeline, or packaging workflow is configured yet. Do not claim those checks passed unless the repository gains an explicit configuration for them.

## Coding Conventions

- Use modern Python type annotations and keep tensor shapes clear at API boundaries.
- Prefer small, explicit data transformations over hidden global state.
- Preserve device and dtype consistency across every tensor in a collated movie batch.
- Validate external data and public model inputs with actionable errors.
- Every function and method must have a brief docstring. Include `Args` and `Returns` sections only when the callable has arguments or returns a value, respectively.
- Begin every docstring with one space immediately after the opening triple quotes, for example `""" Brief description.`
- Keep imports and names on the canonical paths listed above; avoid version suffixes for the supported implementation.

## Testing Expectations

- Add or update tests for every behavior change.
- Data changes should cover filtering, normalization, remapping, hashing, masks, and batching as applicable.
- Model changes should cover tensor shapes, finite outputs or losses, and learned score conditioning when affected.
- Changes to inference should verify that cached and ordinary evaluation-mode predictions remain equivalent.
- Keep tests deterministic by seeding random sampling where exact selections matter.

## Working Boundaries

- Preserve unrelated working-tree changes; this repository may be modified incrementally across sessions.
- Do not commit raw datasets, generated tensor banks, model checkpoints, or secrets unless explicitly requested.
- The inspection scripts are intentionally independent of the model pipeline and should remain usable for evaluating future data sources.
- Treat data fetching, training orchestration, checkpoint management, serving, and UI work as future architecture unless the current task explicitly introduces them.
- Prefer focused changes over broad refactors, and report the exact validation commands run.
