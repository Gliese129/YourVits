from torch import nn
import torch
import math

class PositionEmbedding(nn.Module):
	"""
	Position embedding:
		- pe(pos, 2i)=sin(pos/10000^(2i/d_model))
		- pe(pos, 2i+1)=cos(pos/10000^(2i/d_model))
	
	Attributes:
		pe: (1, max_len, d_model)
	"""
	pe: Tensor

	def __init__(self, d_model: int, max_len=512):
		super().__init__()		
		pe = torch.zeros(max_len, d_model).float()
		pe.requires_grad = False

		position = torch.arange(0, max_len).unsqueeze(1).float()
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)) # use log

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x: Tensor) -> Tensor:
		return self.pe[:, :x.size(1)]
