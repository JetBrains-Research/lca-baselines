from typing import Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer

from src.baselines.backbones.base_backbone import BaseBackbone
from src.baselines.backbones.emb.rankers.base_ranker import BaseRanker
from src.baselines.backbones.emb.rankers.cosine_distance_ranker import CosineDistanceRanker
from src.baselines.backbones.emb.tokenizers.base_tokenizer import BaseTokenizer
from src.baselines.backbones.emb.tokenizers.nltk_tokenizer import NltkTokenizer
from src.baselines.utils.embed_utils import data_to_vectors


class TfIdfEmbBackbone(BaseBackbone):

    def __init__(self,
                 name: str,
                 tokenizer: BaseTokenizer,
                 ranker: BaseRanker,
                 pretrained_path: str):
        super().__init__(name)
        self._tokenizer = tokenizer
        self._ranker = ranker
        self._pretrained_path = pretrained_path

    def localize_bugs(self, issue_description: str, repo_content: Dict[str, str], **kwargs) -> Dict[str, Any]:
        file_names, file_contents = data_to_vectors(issue_description, repo_content)
        self._tokenizer.fit(file_contents)
        model = TfidfVectorizer(tokenizer=self._tokenizer.tokenize)
        vect_file_contents = model.fit_transform(file_contents)

        ranked_file_names, rank_scores = self._ranker.rank(file_names, vect_file_contents)

        return {
            "final_files": list(ranked_file_names),
            "rank_scores": list(rank_scores)
        }
