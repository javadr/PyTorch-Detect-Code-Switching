import numpy as np
import pandas as pd
from collections import Counter


class Data:

    @staticmethod
    def readcsv(file):
        return pd.read_csv(
            f"../data/{file}",
            sep='\t',
            quoting=3,
            encoding='utf8',
            header=None,
            names=['tweet_id', 'user_id', 'start', 'end', 'token', 'label'])

    mktup = lambda df: ["('{token}', '{label}')".format(token=x,label=y) for x,y in zip(df.token, df.label)]
    train = readcsv("train_data.tsv")
    test  = readcsv("dev_data.tsv")
    train['tuple'] = mktup(train)
    test['tuple']  = mktup(test)
    # list of all characters in the vocabulary
    C = set(Counter(''.join(train.token.values.tolist()))) | set(Counter(''.join(test.token.values.tolist())))
    # character embedding layer dimensionality
    d = int(np.log2(len(C))) + 1
    # number of lang id
    labels = list(set(train.label.values)|set(test.label.values))
    mksent = lambda df: df.groupby('tweet_id')['tuple'].apply(','.join).reset_index()
    train_sentences = mksent(train)
    test_sentences  = mksent(test)


