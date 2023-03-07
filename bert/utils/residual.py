from typing import Callable

from torch import nn


class Residual(nn.Module):
    def __init__(self, sublayer: Callable, size: int, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)
        self.sublayer = sublayer

    def forward(self, x, *args, **kwargs):
        x_ = self.norm(x)
        x_ = self.sublayer(x_, *args, **kwargs)
        x_ = self.dropout(x_)
        return x + x_
