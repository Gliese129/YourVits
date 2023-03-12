from torch import nn


class TokenEmbedding(nn.Embedding):
    """a normal token embedding"""
    def __init__(self, vocab_size: int, embed_size=768):
        super().__init__(vocab_size, embed_size, padding_idx=0)
