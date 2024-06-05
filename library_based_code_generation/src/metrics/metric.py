from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def score(self, generated_file: str, reference_code: str, unique_apis: list[str]) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass
