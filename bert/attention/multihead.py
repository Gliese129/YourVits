from torch import nn, Tensor

from .single import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Attributes:
        d_k: dimensions of each head
        h: number of heads
    """
    d_k: int
    h: int

    linear_layers: nn.ModuleList
    output_linear: nn.Module
    dropout: nn.Dropout
    attention: nn.Module

    def __init__(self, h: int, d_model: int, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        b = q.size(0)
        # (b, seq_len, d) -> (b, h, seq_len, d')
        q, k, v = [linear(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
                   for linear, x in zip(self.linear_layers, (q, k, v))]
        x = self.attention(q, k, v, mask=mask, dropout=self.dropout)
        # (b, h, seq_len, d') -> (b, seq_len, d)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        x = self.output_linear(x)
        return x
