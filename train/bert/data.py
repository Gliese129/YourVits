from typing import Callable

import jieba
import MeCab
import nltk
from torch.utils.data import Dataset
from enum import Enum


class Tokenizer(Enum):
    chinese = jieba.lcut
    english = nltk.word_tokenize
    japanese = MeCab.Tagger().parse


class BertDataset(Dataset):
    def __init__(self, data, tokenizer: Callable, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        inputs = self.tokenizer(text)
        return inputs
