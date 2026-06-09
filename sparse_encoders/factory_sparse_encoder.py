# sparse_encoders/factory.py
from sparse_encoders.interface_sparse_encoder import BaseSparseEncoder
# from sparse_encoders.default_encoder import DefaultSparseEncoder


class SparseEncoderFactory:
    """
    Maps user-supplied algorithm name -> BaseSparseEncoder instance.

    Registry keys are lowercase strings (matched case-insensitively).
    Add new encoders here — nowhere else needs to change.

    Usage:
        encoder = SparseEncoderFactory.create("bm25")
        encoder = SparseEncoderFactory.create("splade", model_name="naver/...")
        encoder = SparseEncoderFactory.create(None)   # → DefaultSparseEncoder
    """

    _REGISTRY: dict[str, type] = {}

    @classmethod
    def register(cls, key: str, encoder_cls: type) -> None:
        cls._REGISTRY[key.lower()] = encoder_cls

    @classmethod
    def create(cls, algorithm: str | None, **kwargs) -> BaseSparseEncoder:
        if not algorithm:
            return EndeeBM25(**kwargs)

        key = algorithm.lower().strip()
        encoder_cls = cls._REGISTRY.get(key)
        if encoder_cls is None:
            registered = list(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown sparse algorithm '{algorithm}'. "
                f"Registered options: {registered}"
            )
        return encoder_cls(**kwargs)


# ── Register built-in encoders ────────────────────────────────────────────────
# Import here (not at top) to avoid circular imports and defer heavy deps
from sparse_encoders.concrete_sparse_encoders import EndeeBM25        # noqa: E402

SparseEncoderFactory.register("endee/bm25",   EndeeBM25)