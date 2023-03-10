import argparse

import torch
from torch.utils.data import DataLoader

from bert import Bert
from data import BertDataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', type=int, default=1)
parser.add_argument('-c', '--ctx_len', type=int, default=16)
parser.add_argument('-l', '--max_len', type=int, default=512)
parser.add_argument('-b', '--batch_size', type=int, default=2)
parser.add_argument('-n', '--num_layers', type=int, default=4)
parser.add_argument('-d', '--dropout', type=float, default=0.1)
parser.add_argument('-v', '--vocab_path', type=str, default=None)
parser.add_argument('--data', type=str)
parser.add_argument('--save', type=str, default='./model')
args = parser.parse_args()


epoch = args.epoch
ctx_len = args.ctx_len
offset = ctx_len + 3
max_len = args.max_len
batch_size = args.batch_size
num_layers = args.num_layers
dropout = args.dropout
data_path = args.data
vocab_path = args.vocab_path
save_path = args.save

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = BertDataset(data_path, max_len, ctx_len=ctx_len, vocab=vocab_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
vcb = dataset.vocab
model = Bert(len(vcb), seq_len=max_len, ctx_len=ctx_len, n_layers=num_layers, dropout=dropout)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


losses = []
best_loss = torch.tensor(1e10, dtype=torch.float32)
best_model = model = Bert(len(vcb), seq_len=max_len, ctx_len=ctx_len, n_layers=num_layers, dropout=dropout)


def train(e):
    global best_loss, best_model

    torch.random.manual_seed(e)
    for answers, inputs, pos, segments in tqdm(dataloader):
        ctx = None
        loss = torch.tensor(0., device=device)
        answers, inputs, pos, segments = \
            answers.to(device), inputs.to(device), pos.to(device), segments.to(device)

        optimizer.zero_grad()
        for y, x, p, segment in zip(answers, inputs, pos, segments):

            last_hidden_state, ctx, logits = model(x, segment, context=ctx)

            logits = logits[p].reshape(-1, logits.shape[-1])
            y = y[p].view(-1)
            loss += loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().item())
        if loss < best_loss:
            best_loss = loss
            best_model.load_state_dict(model.state_dict())

    if e % 5 == 0:
        model.save(save_path, f'bert_{e}.pth')


if __name__ == '__main__':
    model = model.to(device)
    best_model = best_model.to(device)

    for e_ in range(epoch):
        train(e_)

    best_model.save(save_path, 'bert_best.pth')
