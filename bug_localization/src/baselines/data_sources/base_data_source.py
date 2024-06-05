from abc import ABC, abstractmethod
from typing import Any, Iterator


class BaseDataSource(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError()
