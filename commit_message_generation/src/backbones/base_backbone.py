from abc import ABC, abstractmethod


class CMGBackbone(ABC):
    name: str = "base"

    @abstractmethod
    def generate_msg(self, preprocessed_commit_mods: str, **kwargs) -> str:
        pass
