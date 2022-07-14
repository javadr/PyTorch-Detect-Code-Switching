import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from config import CFG
# Set random seeds
torch.manual_seed(CFG.seed)


class char2vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.embedding(vocab_size, embed_dim)
