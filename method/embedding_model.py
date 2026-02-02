"""
Embedding models for corpus builder.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from FlagEmbedding import BGEM3FlagModel

@dataclass(frozen=True)
class EmbeddingResult:
    """Standard embedding result format."""
    dense_vecs: np.ndarray
    sparse_vecs: np.ndarray = None


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    def __init__(self, batch_size: int = 32, max_length: int = 8192):
        self.name: str = None
        self.dimension: int = None
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """Encode texts to vectors.

        Args:
            texts: List of text strings to encode

        Returns:
            EmbeddingResult with dense_vecs and optional sparse_vecs

        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")

        result = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length
        )
        return EmbeddingResult(dense_vecs=result['dense_vecs'])


class BGE_M3_Flag(EmbeddingModel):
    """BGE-M3 using FlagEmbedding implementation."""

    def __init__(self, batch_size: int = 32, max_length: int = 8192, dimension: int = 1024, gpu_id: int = 0):
        super().__init__(batch_size=batch_size, max_length=max_length)
        self.name = "bge_m3_flag"
        self.dimension = dimension
        self._model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=True,
            devices=f"cuda:{gpu_id}",
            max_length=self.max_length
        )


class SelfTrain(EmbeddingModel):
    """Self-trained BGE-M3 model from custom path."""

    def __init__(self, model_path: str, batch_size: int = 32, max_length: int = 8192, dimension: int = 1024, gpu_id: int = 0):
        super().__init__(batch_size=batch_size, max_length=max_length)
        self.name = "self_train"
        self.dimension = dimension
        self._model = BGEM3FlagModel(
            model_path,
            use_fp16=True,
            devices=f"cuda:{gpu_id}",
            max_length=self.max_length
        )


def get_model(model_name: str, **kwargs) -> EmbeddingModel:
    """Factory function to create embedding models.

    Args:
        model_name: Name of the model to create
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        An instance of the requested embedding model

    Raises:
        ValueError: If model_name is not supported
    """
    MODELS = {
        "bge_m3_flag": BGE_M3_Flag,
        "self_train": SelfTrain,
    }

    if model_name not in MODELS:
        available_models = list(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")

    model_class = MODELS[model_name]
    return model_class(**kwargs)


def init_embedding_model(model_name: str, model_path: str = None, gpu_id: int = None, batch_size: int = 32) -> EmbeddingModel:
    """Initialize embedding model with common parameters.

    Args:
        model_name: Name of the model to create
        model_path: Path to model weights (required for self_train model)
        gpu_id: GPU device ID
        batch_size: Batch size for encoding

    Returns:
        Initialized embedding model instance

    Raises:
        ValueError: If model_name is "self_train" but model_path is not provided
    """
    model_kwargs = {'batch_size': batch_size}
    if gpu_id is not None:
        model_kwargs['gpu_id'] = gpu_id
    if model_name == "self_train":
        if not model_path:
            raise ValueError("model_path is required for self_train model")
        model_kwargs['model_path'] = model_path
    return get_model(model_name, **model_kwargs)


if __name__ == "__main__":
    model = get_model("bge_m3_flag", gpu_id=6, batch_size=1)
    texts = [
        "Hello World"
    ]
    embeddings = model.encode(texts)
    print(embeddings)