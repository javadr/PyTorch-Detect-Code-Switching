from re import X
import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


from config import CFG

torch.manual_seed(CFG.seed)

@dataclass
class Data:

    @staticmethod
    def readtsv(file):
        df = pd.read_csv(
            f"../data/{file}",
            sep='\t',
            quoting=3,
            encoding='utf8',
            header=None,
            names=['tweet_id', 'user_id', 'start', 'end', 'token', 'label'])
        df['tuple'] = list(zip(df['token'], df['label']))
        tokens = df.groupby('tweet_id')['token'].apply(list).reset_index().set_index('tweet_id')
        labels = df.groupby('tweet_id')['label'].apply(list).reset_index().set_index('tweet_id')
        sentences = pd.concat([tokens, labels], axis=1)

        return df, sentences

    train, train_sentences = readtsv("train_data.tsv")
    test, test_sentences = readtsv("dev_data.tsv")

    # list of all characters in the vocabulary
    Chars = set(Counter(''.join(train.token.values.tolist()))) #| set(Counter(''.join(test.token.values.tolist())))
    # character embedding layer dimensionality
    d = int(np.log2(len(Chars))) + 1


    # list of all tokens and labels(=languages)
    tokens = list(set(train.token.values))
    labels = list(set(train.label.values))

    # token to index and vice versa
    tok2id = {"<PAD>":0, "<UNK>":1}
    tok2id |= {t: i + 2 for i, t in enumerate(tokens)}
    id2tok = {i: t for t,i in tok2id.items()}
    # label to index and vice versa
    lbl2id = {"<PAD>":0}
    lbl2id |= {l: i+1 for i, l in enumerate(labels)}
    id2lbl = {i: l for l,i in lbl2id.items()}
    # character to index and vice versa
    chr2id = {"<PAD>":0, "<UNK>":1}
    chr2id |= {l: i + 2 for i, l in enumerate(Chars)}
    id2chr = {i: l for l,i in chr2id.items()}

    __embedding = lambda dic, col: [[dic.get(t,1) for t in s] for s in col.values]
    X_train_sentences_emb = __embedding(tok2id, train_sentences['token'])
    Y_train_sentences_emb = __embedding(lbl2id, train_sentences['label'])
    X_test_sentences_emb = __embedding(tok2id, test_sentences['token'])
    Y_test_sentences_emb = __embedding(lbl2id, test_sentences['label'])

    @staticmethod
    def decipher_text(encoded_text):
        return ' '.join([Data.id2tok.get(e,1) for e in encoded_text])

    @staticmethod
    def decipher_label(encoded_label):
        return ' '.join([Data.id2lbl.get(e) for e in encoded_label])

    @staticmethod
    def encode_text(text):
        return [[Data.chr2id.get(c,1) for c in token] for token in text.split(' ')]


class CodeSwitchDataset(Dataset):
    def __init__(self, X, Y, train=True):
        self.X = X
        self.Y = Y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.X[idx], self.Y[idx]

train_dataset = CodeSwitchDataset(Data.X_train_sentences_emb, Data.Y_train_sentences_emb)
test_dataset = CodeSwitchDataset(Data.X_test_sentences_emb, Data.Y_test_sentences_emb)

def collate_fn(batch):
    x,y = list(zip(*batch))
    x = [torch.LongTensor(i) for i in x]
    y = [torch.LongTensor(i) for i in y]
    x = pad_sequence(x, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    return x, y

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size,
                        shuffle=True, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=4)

if __name__ == "__main__":
    # for sent, lbl in train_loader:
    #     print(sent.shape, lbl.shape)
    #     print(lbl)
    #     break
    for i in (7,):
        et = Data.X_train_sentences_emb[i]
        el = Data.Y_train_sentences_emb[i]
        print(et, el, Data.decipher_text(et), Data.decipher_label(el), sep='\n')
        print()

    print(Data.encode_text('This is a book !'))