from abc import ABC, abstractmethod
from core.type_registry import FieldSchema

class BaseSparseEncoder(ABC):
    @abstractmethod
    def build_sparse_field(self) -> FieldSchema:
        """Returns a FieldSchema for the sparse field"""

    @abstractmethod
    def encode(self, text: str) -> dict:
        """Returns {"indices": [...], "values": [...]}"""

