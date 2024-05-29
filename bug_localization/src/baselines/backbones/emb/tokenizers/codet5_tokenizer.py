import numpy as np

from transformers import AutoTokenizer

from src.baselines.backbones.emb.tokenizers.base_tokenizer import BaseTokenizer


class CodeT5Tokenizer(BaseTokenizer):

    def __init__(self, checkpoint: str = "Salesforce/codet5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    @staticmethod
    def name():
        return 'codet5'

    def fit(self, file_contents: list[str]):
        pass

    def tokenize(self, file_content: str) -> np.ndarray[str]:
        return self.tokenizer.encode(file_content, return_tensors="pt")[0].numpy()
