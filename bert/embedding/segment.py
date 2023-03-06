from torch import nn


class SegmentEmbedding(nn.Embedding):
    """a learnable embedding for each segment"""
    def __init__(self, embed_size=512):
        super().__init__(4, embed_size, padding_idx=0)
