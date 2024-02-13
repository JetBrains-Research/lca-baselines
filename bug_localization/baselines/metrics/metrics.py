import json
from typing import Optional


class Metrics(object):
    def __init__(self, metrics: Optional[dict] = None):
        self.metrics = metrics if metrics else {}

    @staticmethod
    def from_dict(metrics: dict) -> 'Metrics':
        return Metrics(metrics)

    def to_str(self) -> str:
        return json.dumps(self.metrics)
