from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from baselines.baseline_models import EmbedBaseline
from baselines.baseline_tokenizers import BaseTokenizer


class TfIdfBaseline(EmbedBaseline):

    def __init__(self, pretrained_path: str, tokenizer: BaseTokenizer):
        super().__init__(pretrained_path, tokenizer)

    @staticmethod
    def name():
        return 'tfidf'

    def embed(self, file_contents: List[str]) -> np.ndarray:
        self.tokenizer.fit(file_contents)
        model = TfidfVectorizer(tokenizer=self.tokenizer.tokenize)
        vect_file_contents = model.fit_transform(file_contents)

        print(len(vect_file_contents.toarray()[0]))
        return vect_file_contents.toarray()
