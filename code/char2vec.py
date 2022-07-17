import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np

from config import CFG
from data import Data
# Set random seeds
random.seed(CFG.seed)     # python random generator
np.random.seed(CFG.seed)  # numpy random generator
torch.manual_seed(CFG.seed)


class Char2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeds = nn.Embedding(vocab_size, embed_dim) # first embedding layer for characters
        out_channels1 = 3*4
        out_channels2 = 3*4
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=out_channels1, kernel_size=3),
            nn.ReLU(),
            # nn.Dropout(.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=1),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(out_channels2, out_channels2),
            nn.ReLU(),
        )


    def forward(self, word):
        word = torch.LongTensor([0,*[Data.chr2id[c] for c in word],0])
        embeds = self.embeds(word).transpose(-2,-1)
        conv1 = self.conv1(embeds)
        conv2, _ = self.conv2(conv1).max(dim=-1)
        lin = self.linear(conv2)
        return lin+conv2

c2v = Char2Vec(Data.char_vocab_size, Data.d)
print(c2v('This'))

class BiLSTMtagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size  = embedding_dim,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
            dropout     = 0.5
        )
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_space, dim=1)#.argmax(axis=-1)