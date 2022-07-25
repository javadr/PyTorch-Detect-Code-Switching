import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from config import CFG
from data import Data
# Set random seeds
torch.manual_seed(CFG.seed)
torch.backends.cudnn.deterministic = True


class Char2Vec(nn.Module):
    def __init__(self, char_vocab_size: int, out_ch1: int = CFG.out_ch1, out_ch2: int = CFG.out_ch2):
        super().__init__()
        embed_dim = int(np.log2(char_vocab_size)) + 1
        self.out_ch1, self.out_ch2 = out_ch1, out_ch2
        self.embeds = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0) # first embedding layer for characters
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=out_ch1, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(.1),
        )
        # self.convs1 = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Conv1d(in_channels=embed_dim, out_channels=3*out_ch1, kernel_size=3),
        #             nn.ReLU(),
        #         )
        #         for _ in range(3)
        #     ]
        # )
        # self.drop = nn.Dropout(.25)
        self.convs2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(out_ch1, out_ch2//3, kernel_size=k),
                    nn.ReLU(),
                )
                for k in [3, 4, 5]
            ]
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=out_ch1, out_channels=out_ch2, kernel_size=3),
        #     nn.ReLU(),
        # )
        self.linear = nn.Sequential(
            nn.Linear(out_ch2, out_ch2),
            nn.ReLU(),
        )

    def forward(self, word):
        # word = pad_sequence(torch.LongTensor(word), batch_first=True)
        embeds = self.embeds(word).transpose(-2,-1)
        batch, sent, emb, seq = embeds.shape
        # tmp = [cnn(embeds.view(-1, emb, seq)) for cnn in self.convs1]
        # conv1 = self.drop( torch.cat(tmp, dim=1) )
        conv1 = self.conv1(embeds.view(-1, emb, seq))
        tmp = [cnn(conv1).max(dim=-1)[0].squeeze() for cnn in self.convs2]
        conv2 = torch.cat(tmp, dim=1)
        # conv2, _ = self.conv2(conv1).max(dim=-1)
        lin = self.linear(conv2)
        return (lin+conv2).view(batch, sent, -1)

class BiLSTMtagger(nn.Module):

    def __init__(self, char_vocab_size: int, embedding_dim: int, hidden_dim: int, tagset_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings = Char2Vec(char_vocab_size)
        self.lstm = nn.LSTM(
            input_size  = embedding_dim,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
            dropout     = 0.3
        )
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_space, dim=1)#.argmax(axis=-1)
