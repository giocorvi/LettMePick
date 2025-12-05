import torch

"""
TODO:
    - Add support for variable length sequences (padding + masking)
    - Finish karpathy (GPT from encoder-decoder, transformer lecture from ViTs)
"""

class SelfAttentionHead(torch.nn.Module):
    def __init__(self, input_dim: int, head_dim: int, pdrop: float = 0.2):
        super().__init__()

        self.head_dim = head_dim
        self.key_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.query_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.value_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.dropout = torch.nn.Dropout(pdrop)


    def forward(self, x):
        k, q, v = self.key_head(x), self.query_head(x), self.value_head(x)

        attn = torch.nn.functional.softmax(
            (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5),
            dim=-1
        )

        out = self.dropout(attn) @ v

        return out
    

class MultiSelfAttentionHead(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        head_dim: int,
        attention_pdrop: float = 0.2,
        residual_pdrop: float = 0.2,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attention_heads = torch.nn.ModuleList(
            SelfAttentionHead(input_dim, head_dim, pdrop=attention_pdrop) for _ in range(num_heads)
        )
        self.proj = torch.nn.Linear(num_heads * head_dim, input_dim)
        self.residual_dropout = torch.nn.Dropout(residual_pdrop)

    def forward(self, x):
        # TODO: improve effieciency. Run multiple heads in parallel.
        B, T, _ = x.shape
        out = torch.zeros((B, T, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype)
        for i, head in enumerate(self.attention_heads):
            out[:, :, i] = head(x)

        out = self.proj(out.view((B, T, -1)))

        return self.residual_dropout(out)
        

class FeedForward(torch.nn.Module):
    def __init__(self, embed_dim: int, pdrop: float = 0.2):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 4, embed_dim), # Projection layer
            torch.nn.Dropout(pdrop), # Dropout
        )

    def forward(self, x):
        return self.model(x)


class SelfAttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "The embedding dim has to be a multiple of the number of heads."
        head_dim = embed_dim // num_heads
        self.mh_attention = MultiSelfAttentionHead(
            num_heads=num_heads,
            input_dim=embed_dim,
            head_dim=head_dim,
        )
        self.feed_forward = FeedForward(embed_dim)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.mh_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


# head_dim = ideal_embedding_size // n_head (4?)