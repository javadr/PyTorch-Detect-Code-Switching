from dataclasses import dataclass
@dataclass
class CFG:
    max_sentence_len: int = 50
    batch_size: int = 64
    seed: int = 3407 # seed suggested by arxiv.org/pdf/2109.08203.pdf
    n_epochs: int = 14
    lr: float = 1e-3 # learning rate
    wd: float = 1e-5 # weight dacay
    pad_length: int = 20 # If a word is longer than the pad_length it will be truncated.
    out_ch1: int = 3*4
    out_ch2: int = 5