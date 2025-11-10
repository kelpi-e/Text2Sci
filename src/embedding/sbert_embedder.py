"""Модуль для создания эмбеддингов текста с использованием HuggingFace Transformers."""

from __future__ import annotations
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


class SbertTextEmbedder:
    """Класс для кодирования текстов в векторные эмбеддинги."""

    def __init__(self, model_name: str = "ai-forever/sbert_large_nlu_ru", device: str | None = None):
        print(f"[+] Модель эмбеддингов: {model_name}")
        self.model_name = model_name
        self._model: AutoModel | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._dim: int | None = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        """Ленивая загрузка модели и токенизатора."""
        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._dim = self._model.config.hidden_size

    @staticmethod
    def _mean_pooling(model_output: tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
        """Усреднение эмбеддингов токенов с учётом attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], max_length: int = 128) -> np.ndarray:
        """Кодирует тексты в эмбеддинги."""
        self._load_model()
        assert self._model is not None and self._tokenizer is not None

        if len(texts) == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        encoded_input = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self._model(**encoded_input)

        sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.cpu().numpy()

    def get_embedding_dim(self) -> int:
        """Возвращает размерность эмбеддингов."""
        self._load_model()
        assert self._dim is not None
        return self._dim
