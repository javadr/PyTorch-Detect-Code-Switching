from os import access
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from data import Data, train_dataset, test_dataset, train_loader, test_loader
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

EMBEDDING_DIM = Data.d
HIDDEN_DIM = 128
VOCAB_SIZE = len(Data.tok2id)
TAGSET_SIZE = len(Data.lbl2id)

model = BiLSTMtagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAGSET_SIZE)
loss_function = nn.CrossEntropyLoss()#nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=CFG.lr)
optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=150, factor=0.1, min_lr=1e-8)

flatten = lambda tensor: tensor.view(-1).detach().numpy()

for epoch in tqdm(range(CFG.n_epochs)):

    model.train()  # again, normally you would NOT do 300 epochs, it is toy data
    avg_loss = 0
    train_targets, train_preds = [], []
    for sentence, label in track(train_loader,
                description="Training...", total=len(train_loader), transient=True):
        model.zero_grad()
        scores = model(sentence)
        loss = loss_function(scores.view(-1,scores.shape[-1]), label.view(-1))
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()/len(train_loader)
        train_targets.extend(flatten(label))
        train_preds.extend(flatten(scores.argmax(axis=-1)))

    model.eval()
    avg_val_loss = 0
    val_targets, val_preds = [], []
    for sentence, label in track(test_loader,
                description="Validating...", total=len(test_loader), transient=True):

        scores = model(sentence)
        avg_val_loss += loss_function(scores.view(-1,scores.shape[-1]), label.view(-1)).item()/len(test_loader)
        val_targets.extend(flatten(label))
        val_preds.extend(flatten(scores.argmax(axis=-1)))

    scheduler.step(avg_val_loss)

    #Calculate F1-score
    train_f1 = f1_score(train_targets, train_preds, average='micro')
    val_f1   = f1_score(val_targets,   val_preds,   average='micro')

    width = len(str(CFG.n_epochs))
    print(f'\rEpoch {epoch+1:{width}}/{CFG.n_epochs}, loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}\
    ,train_f1={train_f1:.4f}, val_f1={val_f1:.4f}')

with torch.no_grad():
    out = model(torch.tensor(Data.X_test_sentences_emb[14], dtype=torch.long))
    print( torch.argmax(out, axis=-1).detach().numpy().tolist(),
            Data.Y_test_sentences_emb[14])
