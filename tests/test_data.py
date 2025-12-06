import pandas as pd
import torch

from src.data.data import get_user_ratings_dict


def test_get_user_ratings_dict_warns_and_drops_missing_ids(capsys) -> None:
    raw = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "movie_id": ["m1", "unknown", "m2"],
            "rating_val": [8.0, 9.0, 7.5],
        }
    )
    movie_ids = ["m1", "m2"]

    result = get_user_ratings_dict(raw, movie_ids, lambda x: torch.tensor(x / 10.0))

    captured = capsys.readouterr()
    assert "Warning: Dropping 1 ratings with unknown movie ids out of 3 total." in captured.out

    assert 1 in result
    assert 2 in result
    assert len(result[1]) == 1
    assert len(result[2]) == 1

    idx, rating = result[1][0]
    assert idx == 0
    torch.testing.assert_close(rating, torch.tensor(0.8))

    idx, rating = result[2][0]
    assert idx == 1
    torch.testing.assert_close(rating, torch.tensor(0.75))
