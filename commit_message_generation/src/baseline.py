from typing import Any, Dict, Optional

from .backbones import CMGBackbone
from .preprocessors import CMGPreprocessor
from .utils.typing_utils import UnifiedCommitExample


class CMGBaseline:
    def __init__(self, backbone: CMGBackbone, preprocessor: CMGPreprocessor):
        self._backbone = backbone
        self._preprocessor = preprocessor

    def _preprocess(self, commit: UnifiedCommitExample, **kwargs) -> Dict[str, Any]:
        return self._preprocessor(commit, **kwargs)

    def _predict(self, preprocessed_commit: Dict[str, Any], **kwargs) -> Dict[str, Optional[str]]:
        return self._backbone.generate_msg(preprocessed_commit, **kwargs)

    async def _apredict(self, preprocessed_commit: Dict[str, Any], **kwargs) -> Dict[str, Optional[str]]:
        return await self._backbone.agenerate_msg(preprocessed_commit, **kwargs)

    def generate_msg(self, commit: UnifiedCommitExample, **kwargs) -> Dict[str, Optional[str]]:
        preprocess_kwargs = {kw[len("preprocess_") :]: kwargs[kw] for kw in kwargs if kw.startswith("preprocess_")}
        predict_kwargs = {kw: kwargs[kw] for kw in kwargs if not kw.startswith("preprocess_")}
        preprocessed_commit = self._preprocess(commit, **preprocess_kwargs)
        backbone_output = self._predict(preprocessed_commit, **predict_kwargs)
        return backbone_output

    async def agenerate_msg(self, commit: UnifiedCommitExample, **kwargs) -> Dict[str, Optional[str]]:
        preprocess_kwargs = {kw[len("preprocess_") :]: kwargs[kw] for kw in kwargs if kw.startswith("preprocess_")}
        predict_kwargs = {kw: kwargs[kw] for kw in kwargs if not kw.startswith("preprocess_")}
        preprocessed_commit = self._preprocess(commit, **preprocess_kwargs)
        backbone_output = await self._apredict(preprocessed_commit, **predict_kwargs)
        return backbone_output
