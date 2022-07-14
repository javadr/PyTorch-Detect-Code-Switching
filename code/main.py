import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from data import Data
from config import CFG

torch.manual_seed(CFG.seed)

class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                    batch_first=True, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_space, dim=1)

EMBEDDING_DIM = Data.d
HIDDEN_DIM = 64

model = BiLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(Data.tok2id), len(Data.lbl2id))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    out = model(torch.tensor(Data.X_train_sentences_emb[0], dtype=torch.long))
    print( out, torch.argmax(out, axis=-1),
            Data.Y_train_sentences_emb[0])

for _ in tqdm(range(10)):
    model.train()  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, label in tqdm(zip(Data.X_train_sentences_emb, Data.Y_train_sentences_emb)):
        model.zero_grad()
        scores = model(torch.tensor(sentence, dtype=torch.long))
        loss = loss_function(scores, torch.tensor(label, dtype=torch.long))
        loss.backward()
        optimizer.step()
    model.eval()
    loss = 0
    for sentence, label in tqdm(zip(Data.X_test_sentences_emb, Data.Y_test_sentences_emb)):
        scores = model(torch.tensor(sentence, dtype=torch.long))
        loss = loss_function(scores, torch.tensor(label, dtype=torch.long))


with torch.no_grad():
    out = model(torch.tensor(Data.X_train_sentences_emb[0], dtype=torch.long))
    print( out, torch.argmax(out, axis=-1),
            Data.Y_train_sentences_emb[0])
