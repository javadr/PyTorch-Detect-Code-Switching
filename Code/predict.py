#!/usr/bin/env python3

import argparse
import warnings

import numpy as np
import torch
from rich import print

from config import CFG
from data import Data
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# To make a reproducible output
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)


def predict(model_path: str, text: str):
    model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.eval()
    tokens = tokenizer.tokenize(text)
    x = Data.embedding_s(Data.chr2id, [tokens])
    out = model(torch.LongTensor(x).to(device)).argmax(dim=-1)[0].tolist()
    labels = [Data.id2lbl[i] for i in out]

    return labels[:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Testing a pretrained Character Based CNN+BiLSTM for Code-Switching"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=CFG.saved_models_path / "bestmodel.pth",
        help="path for pre-trained model",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="@Lesambam lmao my sister .. xD",
        help="text string",
    )

    assert (CFG.saved_models_path / "bestmodel.pth").exists(), "Please train the model first!"
    args = parser.parse_args()
    labels = predict(args.model, args.text)

    print(f"input : {args.text}")
    print(f'prediction : {" ".join(labels)}')
