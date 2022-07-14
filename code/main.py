import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

from tqdm import tqdm
from data import Data

class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

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
    print( model(torch.tensor(Data.X_train_sentences_emb[0], dtype=torch.long)) )

for epoch in tqdm(range(3)):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, label in tqdm(zip(Data.X_train_sentences_emb, Data.Y_train_sentences_emb)):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 3. Run our forward pass.
        tag_scores = model(torch.tensor(sentence, dtype=torch.long))

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, torch.tensor(label, dtype=torch.long))
        loss.backward()
        optimizer.step()

with torch.no_grad():
    print( model(torch.tensor(Data.X_train_sentences_emb[0], dtype=torch.long)) )