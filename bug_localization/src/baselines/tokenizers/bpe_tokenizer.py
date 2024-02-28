import os

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from src.baselines.model.baseline_tokenizers import BaseTokenizer


class BPETokenizer(BaseTokenizer):

    def __init__(self,
                 pretrained_path: str,
                 vocab_size=10000,
                 min_frequency=2):
        self.pretrained_path = pretrained_path
        self.tokenizer: Tokenizer
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    @staticmethod
    def name():
        return 'bpe'

    def fit(self, file_contents: list[str]):
        tokenizer_pretrained_path = os.path.join(self.pretrained_path, 'bpe_tokenizer.json')
        if os.path.exists(tokenizer_pretrained_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_pretrained_path)
        else:
            self.tokenizer = Tokenizer(BPE())
            self.tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=self.min_frequency)
            self.tokenizer.train_from_iterator(file_contents, trainer, length=len(file_contents))
            os.makedirs(self.pretrained_path, exist_ok=True)
            self.tokenizer.save(tokenizer_pretrained_path)

    def tokenize(self, file_content: str) -> np.ndarray[str]:
        return self.tokenizer.encode(file_content).tokens
