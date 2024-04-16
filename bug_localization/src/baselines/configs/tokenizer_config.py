from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class TokenizerConfig:
    _target_: str = MISSING


@dataclass
class NltkTokenizerConfig:
    _target_: str = f"src.baselines.backbones.emb.tokenizers.nltk_tokenizer.NltkTokenizer"
