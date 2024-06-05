import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.baselines.backbones.emb.rankers.base_ranker import BaseRanker


class CosineDistanceRanker(BaseRanker):
    def rank(self, file_names: np.ndarray[str], vect_file_contents: np.ndarray[np.ndarray[float]]) \
            -> tuple[np.ndarray[str], np.ndarray[float]]:
        distances = cosine_similarity(vect_file_contents[0].reshape(1, -1), vect_file_contents[1:])[0]
        sorted_indices = np.argsort(distances)[::-1]
        sorted_file_names = file_names[1:][sorted_indices]
        sorted_distances = distances[sorted_indices]

        return sorted_file_names, sorted_distances
