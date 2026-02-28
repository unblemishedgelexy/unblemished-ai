from __future__ import annotations

import asyncio
import hashlib
import threading
from typing import Any


class EmbeddingAdapter:
    def __init__(self, provider: str, dim: int) -> None:
        self._provider = provider.strip().lower()
        self._dim = dim
        self._semantic_model: Any | None = None
        self._semantic_model_lock = threading.Lock()
        self._semantic_available: bool | None = None
        self._semantic_loading = False

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed_sync, text)

    def _embed_sync(self, text: str) -> list[float]:
        if self._provider in _SEMANTIC_PROVIDER_ALIASES:
            semantic = self._embed_sentence_transformers(text)
            if semantic is not None:
                return semantic
            # Preserve semantic vector dimension if model import/runtime fails.
            return _hash_embedding(text=text, dim=_SEMANTIC_EMBEDDING_DIM)

        if self._provider == "hash":
            return _hash_embedding(text=text, dim=self._dim)

        semantic = self._embed_sentence_transformers(text)
        if semantic is not None:
            return semantic
        # Fallback keeps memory pipeline available if semantic model load fails.
        return _hash_embedding(text=text, dim=self._dim)

    def _embed_sentence_transformers(self, text: str) -> list[float] | None:
        model = self._get_or_create_semantic_model()
        if model is None:
            return None

        if not text:
            return [0.0] * _SEMANTIC_EMBEDDING_DIM
        vector = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        raw = vector.tolist() if hasattr(vector, "tolist") else vector
        if not isinstance(raw, list):
            return None
        output: list[float] = []
        for item in raw:
            if isinstance(item, (int, float)):
                output.append(float(item))
        if not output:
            return None

        if len(output) >= _SEMANTIC_EMBEDDING_DIM:
            return output[:_SEMANTIC_EMBEDDING_DIM]
        return output + [0.0] * (_SEMANTIC_EMBEDDING_DIM - len(output))

    def _get_or_create_semantic_model(self) -> Any | None:
        if self._semantic_available is False:
            return None
        if self._semantic_model is not None:
            return self._semantic_model

        with self._semantic_model_lock:
            if self._semantic_available is False:
                return None
            if self._semantic_model is not None:
                return self._semantic_model
            if not self._semantic_loading:
                self._semantic_loading = True
                threading.Thread(
                    target=self._load_semantic_model_sync,
                    daemon=True,
                ).start()
            return None
        return self._semantic_model

    def _load_semantic_model_sync(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(_SEMANTIC_MODEL_NAME)
            with self._semantic_model_lock:
                self._semantic_model = model
                self._semantic_available = True
                self._semantic_loading = False
        except Exception:
            with self._semantic_model_lock:
                self._semantic_available = False
                self._semantic_model = None
                self._semantic_loading = False


def _hash_embedding(text: str, dim: int) -> list[float]:
    if not text:
        return [0.0] * dim

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    seed = digest

    while len(values) < dim:
        for byte in seed:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) >= dim:
                break
        seed = hashlib.sha256(seed).digest()

    norm = sum(v * v for v in values) ** 0.5
    if norm == 0:
        return [0.0] * dim
    return [round(v / norm, 6) for v in values]


_SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
_SEMANTIC_EMBEDDING_DIM = 384
_SEMANTIC_PROVIDER_ALIASES = {
    "sentence_transformers",
    "sentence-transformers",
    "sbert",
    "semantic",
    "minilm",
}
