import json
import math

import jieba
import MeCab
import nltk
import torch
import torchtext.vocab
from torch import Tensor
from torch.utils.data import Dataset

tokenizer = {
    'zh-CN': jieba.lcut,
    'en-US': nltk.word_tokenize,
    'ja-JP': MeCab.Tagger().parse
}


class BertDataset(Dataset):
    def __init__(self, path, max_len, ctx_len=1, vocab=None):
        self.data = None
        self.max_len = max_len
        self.ctx_len = ctx_len
        self.seq_len = max_len - ctx_len - 3
        self.offset = self.ctx_len + 3
        self.vocab = self.load_vocab(vocab) if vocab is not None else \
            torchtext.vocab.vocab({}, specials=['<PAD>', '<MASK>', '<CLS>', '<SEP>', '<UNK>'])
        self.load_data(path)

    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
        self.data = [tokenizer[d['lang']](d['text']) for d in data]
        words = set()
        for sentence in self.data:
            words.update(sentence)
        for word in words:
            self.vocab.append_token(word)

    def load_vocab(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            words = f.readlines()
        self.vocab = torchtext.vocab.vocab({})
        for word in words:
            self.vocab.append_token(word)

    def save_vocab(self, path):
        words = self.vocab.get_stoi().items()
        words = sorted(words, key=lambda x: x[1])
        words = [word[0] for word in words]
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(words))

    def __len__(self):
        return len(self.data)

    def mask(self, inputs: Tensor):
        mask_prob = 0.2
        probs = torch.rand(inputs.size())
        masked = inputs.clone()

        pos = probs < mask_prob
        act_probs = torch.rand(pos.sum())
        replace = inputs[pos]
        replace[act_probs < 0.8] = self.vocab.get_stoi()['<MASK>']
        replace[act_probs >= 0.9] = torch.randint_like(replace[act_probs >= 0.9], low=5, high=len(self.vocab))

        return masked, pos

    def cut(self, seq, part=1):
        max_len = self.seq_len
        pad_idx = self.vocab.get_stoi()['<PAD>']

        if max_len * part < len(seq):
            seq = seq[:max_len * part]

        part_len = math.ceil(len(seq) / part)
        if len(seq) < part_len * part:
            seq += [pad_idx] * (part_len * part - len(seq))
        return [
            seq[i * part_len: (i + 1) * part_len] + [pad_idx] * (max_len - part_len)
            for i in range(part)
        ]

    def __getitem__(self, idx, part=1):
        to_idxes = lambda x: [self.vocab.get_stoi()[w] for w in x]

        text = self.data[idx]
        text = self.vocab(text)
        text = self.cut(text, part=part)

        origin = torch.tensor(text, dtype=torch.long)
        masked, pos = self.mask(origin)

        prefix = to_idxes(['<CLS>', '<SEP>'] + ['<PAD>'] * self.ctx_len + ['<SEP>'])
        prefix = torch.tensor(prefix, dtype=torch.long).unsqueeze(0)
        prefix = torch.repeat_interleave(prefix, origin.size(0), dim=0)

        origin = torch.cat((prefix, origin), dim=-1)
        masked = torch.cat((prefix, masked), dim=-1)
        pos = torch.cat((torch.zeros_like(prefix, dtype=torch.bool), pos), dim=-1)

        segment = torch.zeros_like(origin)
        segment[:, self.offset:] = 1
        return origin, masked, pos, segment
