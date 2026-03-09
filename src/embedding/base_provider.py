from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

