import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from config import CFG
# Set random seeds
torch.manual_seed(CFG.seed)


class char2vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.embedding(vocab_size, embed_dim)

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
            dropout     = 0.25
        )
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_space, dim=1)#.argmax(axis=-1)