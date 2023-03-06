from torch import nn, Tensor

from .attention import MultiHeadAttention
from .embedding import BertEmbedding
from .utils.feed_forward import FeedForward
from .utils.residual import Residual


class BertBlock(nn.Module):
    """
    Bidirectional Encoder:
        Multi-Head Attention -> Add&Norm -> Feed Forward -> Add&Norm
    """
    def __init__(self, d_model: int, attn_heads: int, feed_forward_hidden:int,  dropout: float):
        """

        :param d_model: hidden size
        :param attn_heads: number of attention heads
        :param feed_forward_hidden: hidden size of feed forward network
        :param dropout: dropout rate
        """
        super().__init__()
        self.attention = MultiHeadAttention(h=attn_heads, d_model=d_model)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=feed_forward_hidden, dropout=dropout)

        self.attention = Residual(self.attention, d_model, dropout)
        self.feed_forward = Residual(self.feed_forward, d_model, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.attention(x, x, x, mask=mask)
        x = self.feed_forward(x)
        x = self.dropout(x)
        return x


class Bert(nn.Module):
    """
    Bidirectional Encoder:
        Multi-Head Attention -> Add&Norm -> Feed Forward -> Add&Norm
    """
    def __init__(self, vocab_size, d_model=768, n_layers=4, attn_heads=12, dropout=0.1):
        """

        :param vocab_size: vocab size of total words
        :param d_model: hidden size
        :param n_layers: numbers of Transformer encoder layers
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.embedding = BertEmbedding(vocab_size, d_model, dropout)

        self.bert_blocks = nn.ModuleList([
            BertBlock(d_model, attn_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, segment):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x, segment)
        for block in self.bert_blocks:
            x = block(x, mask)
        return x