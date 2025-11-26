import torch

class CrossAttentionHead(torch.nn.Module):
    def __init__(self, input_dim: int, head_dim: int, pdrop: float = 0.2):
        super().__init__()

        self.head_dim = head_dim
        self.key_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.query_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.value_head = torch.nn.Linear(input_dim, head_dim, bias=False)
        self.dropout = torch.nn.Dropout(pdrop)

    def forward(self, x, y):
        q, k, v = self.query_head(x), self.key_head(y), self.value_head(y)

        attn = torch.nn.functional.softmax(
            (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5),
            dim=-1
        )

        out = self.dropout(attn) @ v

        return out
