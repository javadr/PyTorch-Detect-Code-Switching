#!/usr/bin/env python3

import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data import Data
from config import CFG
from char2vec import BiLSTMtagger

import argparse
from rich import print
import warnings
warnings.filterwarnings("ignore")

# To make a reproducible output
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)

def predict(args):
    model = torch.load(args.model, map_location=torch.device(device))
    model.eval()
    tokens = args.text.split()
    x = Data.embedding_s(Data.chr2id, [tokens])
    out = model(torch.LongTensor(x).to(device)).argmax(dim=-1)[0].tolist()
    labels = [Data.id2lbl[i] for i in out]

    return labels[:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Testing a pretrained Character Based CNN+BiLSTM for Code-Switching")
    parser.add_argument("--model",
                        type=str,
                        default="../saved-models/bestmodel.pth",
                        help="path for pre-trained model")
    parser.add_argument("--text",
                        type=str,
                        default="@Lesambam lmao my sister .. xD",
                        help="text string")

    args = parser.parse_args()
    labels = predict(args)

    print(f'input : {args.text}')
    print(f'prediction : {" ".join(labels)}')
