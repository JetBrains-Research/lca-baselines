import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.baselines.model.baseline_tokenizers import BaseTokenizer
from src.baselines.model.embed_baseline_model import EmbedBaseline


class TfIdfBaseline(EmbedBaseline):

    def __init__(self, repos_path: str, pretrained_path: str, tokenizer: BaseTokenizer):
        super().__init__(repos_path, pretrained_path, tokenizer)

    @staticmethod
    def name():
        return 'tfidf'

    def embed(self, file_content: np.ndarray[str]) -> np.ndarray[np.ndarray[float]]:
        self.tokenizer.fit(file_content)
        model = TfidfVectorizer(tokenizer=self.tokenizer.tokenize)
        vect_file_contents = model.fit_transform(file_content)

        return vect_file_contents.toarray()
