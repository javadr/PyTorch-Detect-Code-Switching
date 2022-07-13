import numpy as np
import pandas as pd
from collections import Counter


class Data:

    @staticmethod
    def readcsv(file):
        return pd.read_csv(
            file,
            sep='\t',
            quoting=3,
            encoding='utf8',
            header=None,
            names=['userid', 'tweetid', 'start', 'end', 'word', 'lang'])

    train = readcsv("data/train_data.tsv")
    test = readcsv("data/dev_data.tsv")
    C = Counter(''.join(train.word.values.tolist()))
    d = int(np.log2(len(C))) + 1  # character embedding layer dimensionality
