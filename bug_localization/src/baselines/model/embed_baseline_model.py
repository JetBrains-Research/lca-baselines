import os
from typing import Optional

import numpy as np
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity

from src.baselines.metrics.classification_metrics import pr_auc_score, roc_auc_score, f1_score
from src.baselines.metrics.metrics import Metrics
from src.baselines.model.baseline_models import Baseline
from src.baselines.model.baseline_tokenizers import BaseTokenizer


class EmbedBaseline(Baseline):

    def __init__(self, repos_path: str, pretrained_path: str, tokenizer: Optional[BaseTokenizer]):
        super().__init__(repos_path)
        self.data_path = repos_path
        self.pretrained_path = pretrained_path
        self.tokenizer = tokenizer

    @staticmethod
    def name() -> str:
        pass

    def embed(self, file_contents: np.ndarray[str]) -> np.ndarray[np.ndarray[float]]:
        pass

    def prepare_data(self, datapoint: dict, category: str) -> tuple[np.ndarray[str], np.ndarray[str], np.ndarray[str]]:
        issue_text = f"{datapoint['issue_title']}\n{datapoint['issue_body']}"
        repo_content = self.get_repo_content(datapoint, category)
        changed_files = self.get_changed_files(datapoint, category)

        file_names = ["issue_text"]
        file_contents = [issue_text]
        for file_name, file_content in repo_content.items():
            file_names.append(file_name)
            file_contents.append(file_name + "\n" + file_content)

        return (np.asarray(file_names, dtype=str),
                np.asarray(file_contents, dtype=str),
                np.asarray(changed_files, dtype=str))

    def run(self, dataset: Dataset, category: str, split: str) -> list[Metrics]:
        metrics_list = []
        for datapoint in dataset:
            file_names, file_contents, changed_files = self.prepare_data(datapoint, category)
            vect_file_contents = self.embed(file_contents)

            y_pred = cosine_similarity(vect_file_contents[0].reshape(1, -1), vect_file_contents[1:])[0]
            y_true = np.isin(file_names[1:], changed_files).astype(int)
            metrics = Metrics(
                {
                    'pr_auc': pr_auc_score(y_true, y_pred),
                    'roc_auc': roc_auc_score(y_true, y_pred),
                    'f1': f1_score(y_true, y_pred),
                    'y_pred': y_pred.tolist(),
                    'y_true': y_true.tolist(),
                    'file_names': file_names[1:].tolist(),
                    'changed_files': int(np.sum(y_true))
                }
            )
            metrics_list.append(metrics)

            print(metrics.to_str())

        return metrics_list

    def get_embeddings_path(self) -> str:
        return os.path.join(self.pretrained_path, self.name(), 'embeddings.npy')

    def dump_embeddings(self, embeddings: np.ndarray[float]):
        np.save(self.get_embeddings_path(), embeddings)

    def load_embeddings(self) -> Optional[np.ndarray[float]]:
        embeddings_path = self.get_embeddings_path()
        if os.path.exists(embeddings_path):
            return np.load(embeddings_path)

        return None
