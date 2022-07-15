from os import access
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from data import Data
from config import CFG
from sklearn.metrics import f1_score

from rich import print
from rich.progress import track

torch.manual_seed(CFG.seed)

class BiLSTMtagger(nn.Module):

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

model = BiLSTMtagger(EMBEDDING_DIM, HIDDEN_DIM, len(Data.tok2id), len(Data.lbl2id))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=CFG.lr)

# with torch.no_grad():
#     out = model(torch.tensor(Data.X_train_sentences_emb[0], dtype=torch.long))
#     print( out, torch.argmax(out, axis=-1),
#             Data.Y_train_sentences_emb[0])

for epoch in tqdm(range(CFG.n_epochs)):
    model.train()  # again, normally you would NOT do 300 epochs, it is toy data
    avg_loss = 0
    train_targets, train_preds = [], []
    for sentence, label in track(zip(Data.X_train_sentences_emb, Data.Y_train_sentences_emb),
                description="Training...", total=len(Data.X_train_sentences_emb), transient=True):
        model.zero_grad()
        scores = model(torch.tensor(sentence, dtype=torch.long))
        loss = loss_function(scores, torch.tensor(label, dtype=torch.long))
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()/len(Data.X_train_sentences_emb)
        train_targets.extend(label)
        train_preds.extend(scores.argmax(axis=-1).detach().numpy())
    model.eval()
    avg_val_loss = 0
    val_targets, val_preds = [], []
    for sentence, label in track(zip(Data.X_test_sentences_emb, Data.Y_test_sentences_emb),
                description="Validating...", total=len(Data.X_test_sentences_emb), transient=True):
        # progress.update(task2, advance=0.5)
        scores = model(torch.tensor(sentence, dtype=torch.long))
        avg_val_loss += loss_function(scores, torch.tensor(label, dtype=torch.long)).item()/len(Data.X_test_sentences_emb)
        val_targets.extend(label)
        val_preds.extend(scores.argmax(axis=-1).detach().numpy())
    # progress.update(task3, advance=0.5)
    #Calculate F1-score
    train_f1 = f1_score(train_targets, train_preds, average='micro')
    val_f1   = f1_score(val_targets,   val_preds,   average='micro')
    width = len(str(CFG.n_epochs))
    print(f'\rEpoch {epoch+1:{width}}/{CFG.n_epochs}, loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}\
    ,train_f1={train_f1:.4f}, val_f1={val_f1:.4f}')

# with torch.no_grad():
#     out = model(torch.tensor(Data.X_test_sentences_emb[14], dtype=torch.long))
#     print( torch.argmax(out, axis=-1),
#             Data.Y_test_sentences_emb[14])
