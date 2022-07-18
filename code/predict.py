
import torch

from data import Data
from config import CFG
from char2vec import BiLSTMtagger

import argparse
from rich import print
import warnings
warnings.filterwarnings("ignore")

EMBEDDING_DIM = 3*CFG.out_ch2
HIDDEN_DIM = 128
TAGSET_SIZE = Data.label_vocab_size # en, es, other
def predict(args):
    model = BiLSTMtagger(EMBEDDING_DIM, HIDDEN_DIM, TAGSET_SIZE)
    state = torch.load(args.model)
    model.load_state_dict(state)
    model.eval()
    tokens = args.text.split()
    x = Data.embedding_s(Data.chr2id, [tokens+['.']])
    out = model(torch.LongTensor(x)).argmax(dim=-1)[0].tolist()
    labels = [Data.id2lbl[i] for i in out]

    return labels[:-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Testing a pretrained Character Based CNN+BiLSTM for Code-Switching")
    parser.add_argument("--model", type=str, default="../saved-models/bestmodel.pth", help="path for pre-trained model")
    parser.add_argument("--text", type=str, default="@lililium This is an audio book !", help="text string")

    args = parser.parse_args()
    labels = predict(args)

    print(f'input : {args.text}')
    print(f'prediction : {" ".join(labels)}')