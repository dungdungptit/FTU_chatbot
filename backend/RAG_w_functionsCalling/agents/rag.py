from typing import *
from abc import ABC, abstractmethod


class RagMini(ABC):
    retriever: Any
    description: str
    output: str = ""

    @abstractmethod
    def get_k_relevant(self, query: str, k: int) -> str:
        raise NotImplementedError()

    @abstractmethod
    def invoke(self, message: str) -> str:
        raise NotImplementedError()
