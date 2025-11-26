import torch

from src.models.movie_encoder import SimpleMovieEncoder


def _build_encoder() -> SimpleMovieEncoder:
    encoder = SimpleMovieEncoder(
        embedding_dim=2,
        num_features=3,
        hidden_size=4,
        output_size=2,
        num_layers=2,
    )

    with torch.no_grad():
        # make the first layer sum all flattened inputs
        encoder.model[0][0].weight.copy_(torch.ones(4, 6))
        encoder.model[0][0].bias.zero_()

        # second layer picks first two hidden dims and adds a bias to the second
        encoder.model[1][0].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            )
        )
        encoder.model[1][0].bias.copy_(torch.tensor([0.0, 1.0]))
    return encoder


def test_movie_encoder_output_shape() -> None:
    encoder = SimpleMovieEncoder(
        embedding_dim=5,
        num_features=4,
        hidden_size=8,
        output_size=3,
        num_layers=1,
    )

    dummy_embeddings = torch.zeros(6, 4, 5)
    output = encoder(dummy_embeddings)

    assert output.shape == (6, 3)


def test_movie_encoder_combines_features_deterministically() -> None:
    encoder = _build_encoder()
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[2.0, 1.0], [0.0, 0.0], [1.0, 2.0]],
        ]
    )

    output = encoder(embeddings)

    expected = torch.tensor([[4.0, 5.0], [6.0, 7.0]])
    torch.testing.assert_close(output, expected)
