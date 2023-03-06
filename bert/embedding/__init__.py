import torch
from torch import nn, Tensor

from .position import PositionEmbedding
from .segment import SegmentEmbedding
from .token import TokenEmbedding


class BertEmbedding(nn.Module):
    """
    Bert embedding takes the sum of the following embeddings:
        - token embedding
        - position embedding
        - segment embedding
    """
    token: TokenEmbedding
    position: PositionEmbedding
    segment: SegmentEmbedding
    dropout: nn.Dropout
    embed_size: int

    def __init__(self, vocab_size: int, embed_size: int, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embed_size)
        self.position = PositionEmbedding(embed_size)
        self.segment = SegmentEmbedding(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size

    def forward(self, sequence: list, label: list) -> Tensor:
        x = torch.sum(
            self.token(sequence),
            self.position(sequence),
            self.segment(label)
        )
        x = self.dropout(x)
        return x
