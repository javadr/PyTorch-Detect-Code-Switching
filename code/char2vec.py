import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Set random seeds
# seed suggested by arxiv.org/pdf/2109.08203.pdf
torch.manual_seed(3407)


class char2vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.embedding(vocab_size, embed_dim)
            