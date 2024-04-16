from dataclasses import dataclass, MISSING


@dataclass
class RankerConfig:
    _target_: str = MISSING


@dataclass
class CosineDistanceRankerConfig:
    _target_: str = f"src.baselines.backbones.emb.rankers.cosine_distance_ranker.CosineDistanceRanker"
