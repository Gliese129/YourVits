import math

import torch.nn.functional as F
from torch import nn, Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    Output = softmax(Q·K^T/sqrt(d_k))·V
    """

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                mask: Tensor = None, dropout: nn.Module = None) -> Tensor:
        # q, k, v: (batch_size, head_num, seq_len, d_k)
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn @ v
