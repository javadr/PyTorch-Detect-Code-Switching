from dataclasses import dataclass
@dataclass
class CFG:
    max_sentence_len: int = 50
    batch_size: int = 64
    seed: int = 3407 # seed suggested by arxiv.org/pdf/2109.08203.pdf
    n_epochs: int = 14
    lr: float = 1e-3 # learning rate
    wd: float = 1e-5 # weight dacay