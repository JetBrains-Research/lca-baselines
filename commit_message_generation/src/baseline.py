from typing import Dict, List, Optional

from .backbones import CMGBackbone
from .preprocessors import CMGPreprocessor
from .utils import CommitDiff


class CMGBaseline:
    def __init__(self, backbone: CMGBackbone, preprocessor: CMGPreprocessor):
        self._backbone = backbone
        self._preprocessor = preprocessor

    def _preprocess(self, commit_mods: List[CommitDiff], **kwargs) -> str:
        return self._preprocessor(commit_mods, **kwargs)

    def _predict(self, preprocessed_commit_mods: str, **kwargs) -> Dict[str, Optional[str]]:
        return self._backbone.generate_msg(preprocessed_commit_mods, **kwargs)

    def generate_msg(self, commit_mods: List[CommitDiff], **kwargs) -> Dict[str, Optional[str]]:
        preprocess_kwargs = {kw[len("preprocess_") :]: kwargs[kw] for kw in kwargs if kw.startswith("preprocess_")}
        predict_kwargs = {kw: kwargs[kw] for kw in kwargs if not kw.startswith("preprocess_")}
        preprocessed_mods = self._preprocess(commit_mods, **preprocess_kwargs)
        backbone_output = self._predict(preprocessed_mods, **predict_kwargs)
        return backbone_output
