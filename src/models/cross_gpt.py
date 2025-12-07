import torch
from .self_gpt import FeedForward

class CrossAttentionHead(torch.nn.Module):
    def __init__(self, input_dim: int, head_dim: int, pdrop: float = 0.2):
        super().__init__()

        self.head_dim = head_dim
        self.key_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.query_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.value_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.dropout = torch.nn.Dropout(pdrop)

    def forward(self, query_states, context_states):
        q, k, v = (
            self.query_head(query_states),
            self.key_head(context_states),
            self.value_head(context_states),
        )

        attn = torch.nn.functional.softmax(
            (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5),
            dim=-1
        )

        out = self.dropout(attn) @ v

        return out
    

class MultiCrossAttentionHead(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        attention_pdrop: float = 0.2,
        residual_pdrop: float = 0.2,
    ):
        super().__init__()

        assert input_dim % num_heads == 0, "The embedding dim has to be a multiple of the number of heads."
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.key_head = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.query_head = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.value_head = torch.nn.Linear(input_dim, input_dim, bias=False)
        
        self.attention_dropout = torch.nn.Dropout(attention_pdrop)
        self.proj = torch.nn.Linear(input_dim, input_dim)
        self.residual_dropout = torch.nn.Dropout(residual_pdrop)

    def forward(self, query_states, context_states):
        B, T_query, _ = query_states.shape
        _, T_ctx, _ = context_states.shape
        q = self.query_head(query_states).reshape((B, T_query, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.key_head(context_states).reshape((B, T_ctx, self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.value_head(context_states).reshape((B, T_ctx, self.num_heads, self.head_dim)).transpose(1, 2)
        
        attn = torch.nn.functional.softmax(
            (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5),
            dim=-1
        )

        out = (self.attention_dropout(attn) @ v).transpose(1, 2).contiguous().reshape((B, T_query, -1))

        out = self.proj(out)

        return self.residual_dropout(out)


class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "The embedding dim has to be a multiple of the number of heads."
        self.mh_attention = MultiCrossAttentionHead(
            num_heads=num_heads,
            input_dim=embed_dim,
        )
        self.feed_forward = FeedForward(embed_dim)
        self.layer_norm_ctx = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_q1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm_q2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, query_states, context_states):
        query_states = query_states + self.mh_attention(
            self.layer_norm_q1(query_states),
            self.layer_norm_ctx(context_states)
        )
        query_states = query_states + self.feed_forward(self.layer_norm_q2(query_states))

        return query_states
