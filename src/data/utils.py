import json
import numbers
from pathlib import Path

import pandas as pd
import torch
from pandas.core.frame import DataFrame
from tqdm import tqdm

from src.data.movie_embedder import FeatureConfig, MovieEmbedder

BASE_DIR = Path(__file__).resolve().parents[2]

def _parse_list(raw_value: str | None) -> list[str]:
    """Return a list of tokens from JSON arrays or comma separated strings."""
    if not raw_value:
        return []
    try:
        value = json.loads(raw_value)
        if isinstance(value, list):
            return [str(v).strip() for v in value]
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def _normalize_feature_value(value) -> str | None:
    """Take the first entry from list-like values and lowercase it."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, str):
        parsed = _parse_list(value)
    elif isinstance(value, (list, tuple, set)):
        parsed = [str(v) for v in value]
    else:
        string_value = str(value).strip()
        return string_value.lower() if string_value else None

    for entry in parsed:
        normalized = str(entry).strip()
        if normalized:
            return normalized.lower()
    return None


def _coerce_numeric(value) -> float | None:
    """Return a numeric scalar if value is an int/float, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real) and not pd.isna(value):
        return float(value)
    return None


def _collect_numeric_values(series: pd.Series) -> list[float]:
    """Extract numeric scalars from a pandas Series."""
    values: list[float] = []
    for value in series:
        numeric = _coerce_numeric(value)
        if numeric is not None:
            values.append(numeric)
    return values


def get_movie_feature_info(
    movie_data: pd.DataFrame, feature_name: str
) -> dict[str, torch.Tensor]:
    """Build a lookup table mapping feature values to ids."""
    if feature_name not in movie_data.columns:
        return {}

    unique_feature_values = set()
    for entry in movie_data[feature_name]:
        normalized = _normalize_feature_value(entry)
        if normalized:
            unique_feature_values.add(normalized)

    if not unique_feature_values:
        unique_feature_values.add("__missing__")

    feature_dict: dict[str, torch.Tensor] = {}
    for i, unique_feature_value in enumerate(sorted(unique_feature_values)):
        feature_dict[unique_feature_value] = torch.tensor(i, dtype=torch.long)

    return feature_dict


def create_dataset(
    movie_data: DataFrame,
    features: list[str],
    embedding_dim: int,
    output_path: Path | None = None,
    max_elements: int | None = None,
    movies_to_consider: list[str] | None = None,
) -> dict:
    """
    Build the embedded movie dataset tensor and persist it to disk.

    Args:
        movie_data: pandas dataframe containing the movie metadata.
        features: ordered list of feature names to include in the embedder.
        embedding_dim: dimensionality for each feature embedding.
        output_path: path to save the serialized dataset.
        max_elements: optional limit on number of movies to encode.
    """
    print("Validating that movie_data is not empty...")
    assert not movie_data.empty, "movie_data cannot be empty"

    print("Validating that at least one feature is provided...")
    assert features, "features list must not be empty"

    features_info: dict[str, dict[str, object]] = {}

    feature_configs = []
    for feat in features:
        print(f"Validating feature '{feat}' exists and has values...")
        assert feat in movie_data.columns, f"Feature '{feat}' missing from dataset"

        numeric_values = _collect_numeric_values(movie_data[feat])
        if numeric_values:
            min_val = min(numeric_values)
            max_val = max(numeric_values)
            if min_val == max_val:
                max_val = min_val + 1e-6
            config = FeatureConfig(name=feat, value_range=(min_val, max_val))
            features_info[feat] = {"numeric": True, "value_range": (min_val, max_val)}
        else:
            lookup = get_movie_feature_info(movie_data, feat)
            assert lookup, f"Feature '{feat}' has no usable values"
            config = FeatureConfig(name=feat, cardinality=len(lookup))
            features_info[feat] = {"numeric": False, "lookup": lookup}
        feature_configs.append(config)

    movie_embedder = MovieEmbedder(
        embedding_dim=embedding_dim,
        feature_configs=feature_configs,
    )

    max_rows = len(movie_data) if max_elements is None else min(
        max_elements, len(movie_data)
    )

    embedded_movies: list[torch.Tensor] = []
    movie_ids: list[str] = []

    progress = tqdm(
        movie_data.iterrows(),
        total=len(movie_data) if max_elements is None else max_elements,
        desc="Embedding movies",
    )

    for _, row in progress:
        if len(movie_ids) >= max_rows:
            break

        movie_id = row.get("movie_id")
        if movie_id is None or (isinstance(movie_id, float) and pd.isna(movie_id)):
            continue

        if movies_to_consider is not None and movie_id not in movies_to_consider:
            continue

        feature_map: dict[str, torch.Tensor] = {}
        for feature in features:
            metadata = features_info[feature]
            if metadata.get("numeric"):
                numeric_value = _coerce_numeric(row.get(feature))
                if numeric_value is None:
                    continue
                feature_map[feature] = torch.tensor(numeric_value, dtype=torch.float32)
                continue

            normalized_value = _normalize_feature_value(row.get(feature))
            if not normalized_value:
                continue
            feature_lookup = metadata["lookup"]
            feature_idx = feature_lookup.get(normalized_value)
            if feature_idx is None:
                continue
            feature_map[feature] = feature_idx.clone().long()

        if not feature_map:
            continue

        with torch.no_grad():
            embedded = movie_embedder([feature_map])
        embedded_movies.append(embedded.squeeze(0))
        movie_ids.append(str(movie_id))

    if not movie_ids:
        raise ValueError("No movies matched the provided features")

    embedded_movies = torch.stack(embedded_movies).detach().clone()

    dataset = {
        "data": embedded_movies,
        "ids": movie_ids,
        "feature_configs": [cfg.__dict__ for cfg in feature_configs],
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, output_path)

    return dataset

if __name__ == "__main__":
    data = pd.read_csv('data/ratings_subset.csv', lineterminator='\n')
    movies_to_consider = []
    for d in data.itertuples(index=True):
        movies_to_consider.append(d.movie_id)

    movie_data_path = BASE_DIR / "data" / "movie_data.csv"
    if movie_data_path.exists():
        df = pd.read_csv(movie_data_path, lineterminator='\n')
        create_dataset(
            movie_data=df,
            features=["genres", "movie_id", "production_countries", "year_released"],
            embedding_dim=6,
            output_path=BASE_DIR / "data" / "subset_movie_dataset-id-pc-yr.pt",
            # max_elements=5000,
            movies_to_consider=movies_to_consider,
        )
