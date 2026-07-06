# sparse_encoders/endee_bm25.py
from endee_model import SparseModel
from sparse_encoders.interface_sparse_encoder import BaseSparseEncoder
from core.schema import FieldSchema, FieldType, FieldRole


class EndeeBM25(BaseSparseEncoder):
    """
    BM25 sparse encoder backed by endee-model's SparseModel.
    SparseModel.embed() is a batch API — always pass a list.
    """

    def __init__(self, model_name: str = "endee/bm25"):
        self._model = SparseModel(model_name=model_name)

    def build_sparse_field(self) -> FieldSchema:
        return FieldSchema(
            name="sparse_vector",
            field_type=FieldType.SPARSE_VECTOR,
            role=FieldRole.SPARSE_VECTOR,
        )

    def encode(self, text: str) -> dict:
        # embed() is a batch API — wrap in list, take first result
        result = list(self._model.embed([text]))[0]
        return {
            "indices": result.indices.tolist(),
            "values":  result.values.tolist(),
        }

    def encode_batch(self, texts: list[str]) -> list[dict]:
        """
        Preferred for batch migration — avoids repeated single-item calls.
        Returns list of {"indices": [...], "values": [...]} aligned with input.
        """
        results = list(self._model.embed(texts))
        return [
            {
                "indices": r.indices.tolist(),
                "values":  r.values.tolist(),
            }
            for r in results
        ]
