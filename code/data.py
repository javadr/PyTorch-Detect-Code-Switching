import re
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

    def readtsv(file):
        df = pd.read_csv(
            f"../data/{file}",
            sep='\t',
            quoting=3,
            encoding='utf8',
            header=None,
            names=['tweet_id', 'user_id', 'start', 'end', 'token', 'label'])
        df['token'] = df['token'].apply(lambda t: re.sub(r'(.)\1{4,}',r'\1\1\1\1', t)[:CFG.pad_length])
        df['tuple'] = list(zip(df['token'], df['label']))
        tokens = df.groupby('tweet_id')['token'].apply(list).reset_index().set_index('tweet_id')
        labels = df.groupby('tweet_id')['label'].apply(list).reset_index().set_index('tweet_id')
        sentences = pd.concat([tokens, labels], axis=1).rename(lambda c: c+'s', axis='columns')

        return df, {'tokens': sentences['tokens'].values, 'labels': sentences['labels'].values}

    train, train_sentences = readtsv("train_data.tsv")
    test, test_sentences = readtsv("dev_data.tsv")

    # list of all characters in the vocabulary
    Chars = sorted(set(Counter(''.join(train.token.values.tolist())))) #| set(Counter(''.join(test.token.values.tolist())))
    # character embedding layer dimensionality
    d = int(np.log2(len(Chars))) + 1


    # list of all tokens and labels(=languages)
    tokens = sorted(set(train.token.values))
    labels = sorted(set(train.label.values))

    # token to index and vice versa
    tok2id = {"<PAD>":0, "<UNK>":1, "<S>":2, "</S>":3}
    tok2id.update({t: i + 4 for i, t in enumerate(tokens)})
    id2tok = {i: t for t,i in tok2id.items()}
    # label to index and vice versa
    lbl2id = {"<PAD>":0}
    lbl2id.update({l: i+1 for i, l in enumerate(labels)})
    id2lbl = {i: l for l,i in lbl2id.items()}
    # character to index and vice versa
    chr2id = {"<PAD>":0, "<UNK>":1, "<S>":2, "</S>":3}
    chr2id.update({l: i + 4 for i, l in enumerate(Chars)})
    id2chr = {i: l for l,i in chr2id.items()}

    # vocabulary size
    char_vocab_size = len(chr2id)
    token_vocab_size = len(tok2id)
    label_vocab_size = len(lbl2id)


    embedding_s = lambda dic, data: [[ [dic["<S>"]]+[dic.get(c,1) for c in w]+[dic["</S>"]]\
            +[0]*(CFG.pad_length-len(w)+1) for w in sent ] for sent in data ]
    embedding = lambda dic, data: [[dic.get(t,1) for t in s] for s in data]

    X_train_sentences_emb = embedding(tok2id, train_sentences['tokens'])
    X_test_sentences_emb  = embedding(tok2id, test_sentences['tokens'])
    # X_train_sentences_emb = __embedding_s(chr2id, train_sentences['tokens'])
    # X_test_sentences_emb  = __embedding_s(chr2id, test_sentences['tokens'])
    Y_train_sentences_emb = embedding(lbl2id, train_sentences['labels'])
    Y_test_sentences_emb  = embedding(lbl2id, test_sentences['labels'])

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

# train_dataset = CodeSwitchDataset(Data.X_train_sentences_emb, Data.Y_train_sentences_emb)
# test_dataset = CodeSwitchDataset(Data.X_test_sentences_emb, Data.Y_test_sentences_emb)
train_dataset = CodeSwitchDataset(Data.train_sentences['tokens'], Data.train_sentences['labels'])
test_dataset = CodeSwitchDataset(Data.test_sentences['tokens'], Data.test_sentences['labels'])

def word_encode(word):
    x = torch.zeros(CFG.pad_length)

def collate_fn(batch):
    x,y = list(zip(*batch)) # makes all sentences into x and all labels into y
    x,y = Data.embedding_s(Data.chr2id, x), Data.embedding(Data.lbl2id, y)
    x = [torch.LongTensor(i) for i in x]
    y = [torch.LongTensor(i) for i in y]
    x = pad_sequence(x, batch_first=True)
    y = pad_sequence(y, batch_first=True)
    return x, y

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size,
                        shuffle=True, collate_fn=collate_fn, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)

if __name__ == "__main__":
    # for sent, lbl in train_loader:
    #     print(sent.shape, lbl.shape)
    #     print(lbl)
    #     break

    # for i in (7,14):
    #     et = Data.X_test_sentences_emb[i]
    #     el = Data.Y_test_sentences_emb[i]
    #     print(et, el, Data.decipher_text(et), Data.decipher_label(el), sep='\n')
    #     print()
    # print(Data.encode_text('This is a book !'))

    for sent, lab in train_loader:
        print(sent.shape, lab.shape, end=' ')
        exit()
    # print('\n')
    # for sent, lab in test_loader:
    #     print(lab.shape[1], end=' ')