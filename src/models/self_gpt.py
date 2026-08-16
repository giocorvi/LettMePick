""" Self-attention layers used by movie and user-context encoders."""

import torch

class SelfAttentionHead(torch.nn.Module):
    def __init__(self, input_dim: int, head_dim: int, pdrop: float = 0.2):
        """ Initialize a single self-attention head.

        Args:
            input_dim: Width of each input token.
            head_dim: Width of projected keys, queries, and values.
            pdrop: Attention dropout probability.
        """
        super().__init__()

        self.head_dim = head_dim
        self.key_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.query_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.value_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.dropout = torch.nn.Dropout(pdrop)


    def forward(self, x):
        """ Apply self-attention to a token sequence.

        Args:
            x: Input tensor shaped ``[batch, tokens, input_dim]``.

        Returns:
            Attended tokens for this head.
        """
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
        attention_pdrop: float = 0.2,
        residual_pdrop: float = 0.2,
    ):
        """ Initialize optimized multi-head self-attention.

        Args:
            num_heads: Number of parallel attention heads.
            input_dim: Shared input and output width.
            attention_pdrop: Dropout probability for attention weights.
            residual_pdrop: Dropout probability for projected outputs.
        """
        super().__init__()

        assert input_dim % num_heads == 0, "The embedding dim has to be a multiple of the number of heads."
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.key_head = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.query_head = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.value_head = torch.nn.Linear(input_dim, input_dim, bias=False)

        self.proj = torch.nn.Linear(input_dim, input_dim)
        self.attention_dropout = torch.nn.Dropout(attention_pdrop)
        self.residual_dropout = torch.nn.Dropout(residual_pdrop)

    def forward(self, x):
        """ Apply multi-head self-attention to a token sequence.

        Args:
            x: Input tensor shaped ``[batch, tokens, input_dim]``.

        Returns:
            Attended tokens with the original embedding width.
        """
        # Project once at full width, then split channels into independent heads.
        B, T, _ = x.shape
        k = self.key_head(x).reshape((B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        q = self.query_head(x).reshape((B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.value_head(x).reshape((B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        
        attn = torch.nn.functional.softmax(
            (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5),
            dim=-1
        )

        out = (self.attention_dropout(attn) @ v).transpose(1, 2).contiguous().reshape((B, T, -1))

        out = self.proj(out)

        return self.residual_dropout(out)


class MaskedMultiSelfAttentionHead(MultiSelfAttentionHead):
    def forward(self, x, mask: torch.Tensor):
        """ Apply self-attention while ignoring padded key positions.

        Args:
            x: Input tensor shaped ``[batch, tokens, input_dim]``.
            mask: Boolean validity mask shaped ``[batch, tokens]``.

        Returns:
            Attended tokens with the original embedding width.
        """
        B, T, _ = x.shape
        k = self.key_head(x).reshape((B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        q = self.query_head(x).reshape((B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.value_head(x).reshape((B, T, self.num_heads, self.head_dim)).transpose(1, 2)

        # Mask padded keys so valid tokens cannot retrieve padded metadata.
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_scores = attn_scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = torch.nn.functional.softmax(attn_scores, dim=-1)

        out = (self.attention_dropout(attn) @ v).transpose(1, 2).contiguous().reshape((B, T, -1))
        out = self.proj(out)
        return self.residual_dropout(out)
        

class FeedForward(torch.nn.Module):
    def __init__(self, embed_dim: int, pdrop: float = 0.2):
        """ Initialize a transformer feed-forward sublayer.

        Args:
            embed_dim: Input and output embedding width.
            pdrop: Output dropout probability.
        """
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 4, embed_dim), # Projection layer
            torch.nn.Dropout(pdrop), # Dropout
        )

    def forward(self, x):
        """ Transform each token independently.

        Args:
            x: Token embeddings with width ``embed_dim``.

        Returns:
            Transformed token embeddings.
        """
        return self.model(x)


class SelfAttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        """ Initialize a residual self-attention block.

        Args:
            embed_dim: Input and output embedding width.
            num_heads: Number of parallel attention heads.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "The embedding dim has to be a multiple of the number of heads."
        self.mh_attention = MultiSelfAttentionHead(
            num_heads=num_heads,
            input_dim=embed_dim,
        )
        self.feed_forward = FeedForward(embed_dim)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        """ Apply normalized attention and feed-forward residuals.

        Args:
            x: Input token embeddings.

        Returns:
            Contextualized token embeddings.
        """
        # Pre-normalized attention and feed-forward updates preserve residual state.
        x = x + self.mh_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class MaskedSelfAttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        """ Initialize a residual masked self-attention block.

        Args:
            embed_dim: Input and output embedding width.
            num_heads: Number of parallel attention heads.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "The embedding dim has to be a multiple of the number of heads."
        self.mh_attention = MaskedMultiSelfAttentionHead(
            num_heads=num_heads,
            input_dim=embed_dim,
        )
        self.feed_forward = FeedForward(embed_dim)
        self.layer_norm_1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """ Apply residual self-attention while ignoring padded keys.

        Args:
            x: Input token embeddings.
            mask: Boolean validity mask for token positions.

        Returns:
            Contextualized token embeddings.
        """
        # Use the same pre-norm residual structure while masking metadata padding.
        x = x + self.mh_attention(self.layer_norm_1(x), mask)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
