import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


class Data:

    @staticmethod
    def readtsv(file):  # sourcery skip: identity-comprehension
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
    lbl2id |= {l: i + 1 for i, l in enumerate(labels)}
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
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
