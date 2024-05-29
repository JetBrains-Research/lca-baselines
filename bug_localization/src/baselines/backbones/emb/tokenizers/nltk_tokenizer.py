import string
import re

import nltk
import numpy as np
from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from src.baselines.backbones.emb.tokenizers.base_tokenizer import BaseTokenizer


class NltkTokenizer(BaseTokenizer):
    name = 'nltk'

    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

    @staticmethod
    def _camel_case_split(token: str) -> list[str]:
        matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", token)
        return [m.group(0) for m in matches]

    def fit(self, file_contents: list[str]):
        pass

    def tokenize(self, file_content: str) -> np.ndarray[str]:
        tokens = word_tokenize(file_content)
        stop_words = set(stopwords.words("english"))

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        prep_tokens: list[str] = []
        for token in tokens:
            if token in string.punctuation:
                continue
            sub_tokens = self._camel_case_split(token)
            for sub_token in sub_tokens:
                prep_token = lemmatizer.lemmatize(stemmer.stem(sub_token.lower()))
                if prep_token not in stop_words:
                    prep_tokens.append(prep_token)

        return np.array(prep_tokens)
