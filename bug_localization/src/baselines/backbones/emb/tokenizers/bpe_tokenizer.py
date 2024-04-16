import os
from typing import Optional

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from src.baselines.backbones.emb.tokenizers.base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    name = 'bpe'

    def __init__(self, pretrained_path: Optional[str] = None, vocab_size=10000, min_frequency=2):
        self.pretrained_path = pretrained_path
        self.tokenizer: BaseTokenizer
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def fit(self, file_contents: list[str]):
        if os.path.exists(self.pretrained_path):
            # TODO: read from file
            pass
        else:
            self.tokenizer = Tokenizer(BPE())
            trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=self.min_frequency)
            self.tokenizer.train_from_iterator(file_contents, trainer, length=len(file_contents))
            os.makedirs(self.pretrained_path, exist_ok=True)
            self.tokenizer.save(self.pretrained_path)

    def tokenize(self, file_content: str) -> np.ndarray[str]:
        return self.tokenizer.encode(file_content).tokens
