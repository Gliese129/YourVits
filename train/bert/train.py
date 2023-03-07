import torch

from bert import Bert

if __name__ == '__main__':
    model = Bert(100)
    x = torch.randint(0, 100, (2, 512))
    segment = torch.randint(0, 3, (2, 512))
    context = torch.rand((2, 512, 768))
    y, ctx = model(x, segment, context=context)
    print(y.shape, ctx.shape)
    print(y, ctx)
    loss = (1 - y).sum()
    loss.backward()
    print(model.embedding.token.weight.grad)
