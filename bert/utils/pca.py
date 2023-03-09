from sklearn.decomposition import PCA
import torch
from torch import nn, Tensor


class PCALayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.pca = PCA(n_components=output_size)

    def forward(self, x: Tensor):
        # use PCA to reduce context length
        # (batch, seq_len, h) -> (batch, h, seq_len) -> (batch, h, context_len) -> (batch, context_len, h)
        b, s, h = x.shape
        dtype, device = x.dtype, x.device
        x = x.transpose(1, 2).reshape(b * h, -1)
        x = self.pca.fit_transform(x.detach().cpu().numpy())
        x = torch.tensor(x, dtype=dtype, device=device)
        x = x.reshape(b, h, -1).transpose(1, 2)

        return x

    def backward(self, grad_output):
        b, s, h = grad_output.shape
        dtype, device = grad_output.dtype, grad_output.device
        # 将梯度reshape成2D tensor
        grad_output = grad_output.transpose(1, 2).reshape(b * s, -1)
        # 调用sklearn的PCA的inverse_transform方法
        grad_output = self.pca.inverse_transform(grad_output.detach().cpu().numpy())
        # 将梯度reshape回3D tensor
        grad_output = torch.tensor(grad_output, dtype=dtype, device=device).reshape(b, s, h)
        grad_output = grad_output.transpose(1, 2)

        return grad_output
