#!/usr/bin/env python3

from pathlib import Path
from dataclasses import dataclass


@dataclass
class CFG:
    max_sentence_len: int = 50
    batch_size: int = 64
    seed: int = 3407  # seed suggested by http://arxiv.org/pdf/2109.08203.pdf
    n_epochs: int = 100
    lr: float = 1e-3  # learning rate
    wd: float = 1e-5  # weight decay
    pad_length: int = 20  # If a word is longer than the pad_length it will be truncated.
    out_ch1: int = 3 * 7
    out_ch2: int = (
        3 * 5
    )  # dividable by 3, as we have 3 different kernel sizes concatenated together
    working_dir: Path = Path(__file__).parent.parent
    data_path: Path = working_dir / "Data"
    result_path: Path = working_dir / "Results"
    saved_models_path: Path = result_path / "saved-models"
    images_path: Path = result_path / "images"


# To ensure that the required directories are created.
for dir in (CFG.saved_models_path, CFG.images_path):
    dir.mkdir(exist_ok=True, parents=True)
