import torch
from torch.utils.data import DataLoader

from bert import Bert
from data import BertDataset
from tqdm import tqdm

dataset = BertDataset('./data/data.json', 512)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
vcb = dataset.vocab
model = Bert(len(vcb))

epoch = 1
ctx_len = 1
offset = ctx_len + 3

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


def train(e):
    torch.random.manual_seed(e)
    for answers, inputs, pos, segments in tqdm(dataloader):
        ctx = None
        loss = torch.tensor(0.)
        optimizer.zero_grad()
        for y, x, p, segment in zip(answers, inputs, pos, segments):

            last_hidden_state, ctx, logits = model(x, segment, context=ctx)

            logits = logits[p].reshape(-1, logits.shape[-1])
            y = y[p].view(-1)
            loss += loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        print(loss)


if __name__ == '__main__':
    for e_ in range(epoch):
        train(e_)
