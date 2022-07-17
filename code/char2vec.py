import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from config import CFG
from data import Data, train_loader
# Set random seeds
torch.manual_seed(CFG.seed)
torch.backends.cudnn.deterministic = True


class Char2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim, out_ch1= CFG.out_ch1, out_ch2= CFG.out_ch2):
        super().__init__()
        self.out_ch1, self.out_ch2 = out_ch1, out_ch2
        self.embeds = nn.Embedding(vocab_size, embed_dim) # first embedding layer for characters
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=out_ch1, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_ch1, out_channels=out_ch2, kernel_size=3),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(out_ch2, out_ch2),
            nn.ReLU(),
        )

    def forward(self, word):
        # word = pad_sequence(torch.LongTensor(word), batch_first=True)
        embeds = self.embeds(word).transpose(-2,-1)
        batch, sent, emb, seq = embeds.shape
        conv1 = self.conv1(embeds.view(-1, emb, seq))
        conv2, _ = self.conv2(conv1).max(dim=-1)
        lin = self.linear(conv2)
        return (lin+conv2).view(batch, sent, -1)

if __name__=="__main__":
    c2v = Char2Vec(Data.char_vocab_size, Data.d)
    for sent, lab in train_loader:
        em = c2v(sent)
        print(em.shape)
        print(em[0])
        exit()
    # print(c2v([torch.LongTensor([0, *[Data.chr2id[c] for c in t],0]).transpose(-1,0) for t in ['This', 'is']]))

class BiLSTMtagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings = Char2Vec(Data.char_vocab_size, Data.d)
        self.lstm = nn.LSTM(
            input_size  = embedding_dim,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
            dropout     = 0.25
        )
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_space, dim=1)#.argmax(axis=-1)