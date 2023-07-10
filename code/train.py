#!/usr/bin/env python3

import warnings
from collections import defaultdict
from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rich import print
from rich.progress import track

from utils import *
from config import CFG
from char2vec import BiLSTMtagger
from data import Data, test_loader, train_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")

# To make a reproducible output
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)

EMBEDDING_DIM = CFG.out_ch2
HIDDEN_DIM = 128
TARGET_SIZE = Data.label_vocab_size  # en, es, other
CHAR_VOCAB_SIZE = Data.char_vocab_size

model = BiLSTMtagger(CHAR_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM,
                    TARGET_SIZE).to(device)
loss_function = nn.CrossEntropyLoss(
    ignore_index=Data.label_vocab_size)  #nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=CFG.lr)
optimizer = optim.Adam(
    model.parameters(),
    lr=CFG.lr,
    weight_decay=CFG.wd,
)
scheduler = ReduceLROnPlateau(
    optimizer,
    'min',
    patience=150,
    factor=0.1,
    min_lr=1e-8
)

logs = defaultdict(list)

best_val_score = 0

width = len(str(CFG.n_epochs))
for epoch in (range(CFG.n_epochs + 1)):

    model.train()  # again, normally you would NOT do 300 epochs, it is toy data
    avg_loss = 0
    train_targets, train_preds = [], []
    if epoch != 0:
        for sentence, label, sent_lens in track(
                train_loader,
                description="Training...",
                total=len(train_loader),
                transient=True,
            ):
            model.zero_grad()
            scores = model(sentence)
            loss = loss_function(
                scores.view(-1, scores.shape[-1]),
                label.view(-1)
            )
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            train_targets.extend(flatten(label, sent_lens))
            train_preds.extend(flatten(scores.argmax(axis=-1), sent_lens))

    model.eval()
    avg_val_loss = 0
    val_targets, val_preds = [], []
    for sentence, label, sent_lens in track(
            test_loader,
            description="Validating...",
            total=len(test_loader),
            transient=True,
        ):
        scores = model(sentence)
        avg_val_loss += loss_function(
            scores.view(-1, scores.shape[-1]),
            label.view(-1)
            ).item() / len(test_loader)
        val_targets.extend(flatten(label, sent_lens))
        val_preds.extend(flatten(scores.argmax(axis=-1), sent_lens))

    scheduler.step(avg_val_loss)

    #Calculate F1-score, accuracy_score
    train_eval = evaluation(train_targets,
                            train_preds,
                            metrics=['f1', 'accuracy'])
    val_eval = evaluation(val_targets, val_preds, metrics=['f1', 'accuracy'])

    if epoch <= 5 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:{width}}/{CFG.n_epochs}, loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}",
            f",f1={train_eval['f1']:.4f}, val_f1={val_eval['f1']:.4f}",
            f",,acc={train_eval['accuracy']:.4f}, val_acc={val_eval['accuracy']:.4f}"
        )

    if epoch != 0:
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

torch.save(
    best_model,
    f'../saved-models/model-[{datetime.now().strftime("%y%m%d%H%M")}]{max(logs["val_f1"]):.5f}.pth'
    .replace('0.', '.'))
torch.save(best_model, '../saved-models/bestmodel.pth')

res_plot(logs, desc="BiLSTM+Char2Vec, 2Layers, Adam, lre-3, wde-5")

cls_report(best_model)
