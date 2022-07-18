import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import Data, train_loader, test_loader
from config import CFG
from utils import *
from char2vec import BiLSTMtagger

from tqdm import tqdm
from rich import print
from rich.progress import track

from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(CFG.seed)

EMBEDDING_DIM = 3*CFG.out_ch2
HIDDEN_DIM = 128
TAGSET_SIZE = Data.label_vocab_size # en, es, other

model = BiLSTMtagger(EMBEDDING_DIM, HIDDEN_DIM, TAGSET_SIZE)
loss_function = nn.CrossEntropyLoss()#nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=CFG.lr)
optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=150, factor=0.1, min_lr=1e-8)

logs = defaultdict(list)

best_val_score = 0

for epoch in (range(CFG.n_epochs+1)):

    model.train()  # again, normally you would NOT do 300 epochs, it is toy data
    avg_loss = 0
    train_targets, train_preds = [], []
    if epoch!=0:
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

    #Calculate F1-score, accuracy_score
    train_eval = evaluation(train_targets, train_preds, metrics=['f1', 'accuracy'])
    val_eval   = evaluation(val_targets,   val_preds,   metrics=['f1', 'accuracy'])

    width = len(str(CFG.n_epochs))
    print(f"Epoch {epoch:{width}}/{CFG.n_epochs}, loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}\
 ,f1={train_eval['f1']:.4f}, val_f1={val_eval['f1']:.4f}\
 ,acc={train_eval['accuracy']:.4f}, val_acc={val_eval['accuracy']:.4f}")
    if epoch!=0:
        logs['train_loss'].append(avg_loss)
        logs['val_loss'].append(avg_val_loss)
        logs['train_f1'].append(train_eval['f1'])
        logs['val_f1'].append(val_eval['f1'])
        logs['train_accuracy'].append(train_eval['accuracy'])
        logs['val_accuracy'].append(val_eval['accuracy'])
    # saving the best model
    if best_val_score < val_eval['f1']:
        best_val_score = val_eval['f1']
        best_model = model

torch.save(model, f"../saved-models/model-{max(logs['val_f1']):.5f}.pth".replace('0.','.'))

res_plot(logs, desc="BiLSTM+Char2Vec, 2Layers, Adam, lre-3, wde-5")
