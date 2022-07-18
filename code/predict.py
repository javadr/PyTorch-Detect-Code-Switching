#!/usr/bin/env python3

import argparse

import torch
from data import Data
from rich import print

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()


def predict(args):

    model = torch.load(args.model)
    model.eval()
    if use_cuda:
        model = model.to("cuda")
    tokens = args.text.split()
    x = Data.embedding_s(Data.chr2id, [tokens+['.']])
    out = model(torch.LongTensor(x)).argmax(dim=-1)[0].tolist()
    labels = [Data.id2lbl[i] for i in out]

    return labels[:-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Testing a pretrained Character Based CNN for Code-Switching")
    parser.add_argument("--model", type=str, default="../saved-models/bestmodel.pth", help="path for pre-trained model")
    parser.add_argument("--text", type=str, default="@lililium This is an audio book !", help="text string")

    args = parser.parse_args()
    labels = predict(args)

    print(f'input : {args.text}')
    print(f'prediction : {" ".join(labels)}')