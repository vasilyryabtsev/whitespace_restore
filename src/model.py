import torch
import torch.nn as nn

from transformers import T5EncoderModel, AutoTokenizer
from typing import Optional


class ByT5WhitespaceRestorer(nn.Module):
    """
    ByT5 модель для восстановления пробелов в тексте.

    Использует предобученный ByT5-small энкодер и добавляет голову классификации
    для предсказания места вставки пробелов (0 - не добавлять, 1 - добавить пробел).
    """

    def __init__(self, dropout_rate: float = 0.1, freeze_encoder: bool = False):
        """
        Инициализация модели.

        Args:
            dropout_rate: Коэффициент dropout для классификационной головы
            freeze_encoder: Заморозить ли веса энкодера
        """
        super().__init__()

        # Загрузка предобученного ByT5
        model_name = "google/byt5-small"
        self.encoder = T5EncoderModel.from_pretrained(model_name)

        # Заморозка весов энкодера если нужно
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Размерность скрытого состояния энкодера
        self.hidden_size = self.encoder.config.d_model

        # Классификационная голова
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(
                self.hidden_size, 2
            ),  # 2 класса: 0 - не добавлять, 1 - добавить пробел
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Прямой проход модели.

        Args:
            input_ids: Токенизированный входной текст [batch_size, seq_len]
            attention_mask: Маска внимания [batch_size, seq_len]
            labels: Целевые метки для обучения [batch_size, seq_len]

        Returns:
            logits: Логиты предсказаний
        """
        # Получение скрытых представлений от энкодера
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Последний скрытый слой [batch_size, seq_len, hidden_size]
        hidden_states = encoder_outputs.last_hidden_state

        # Применение классификационной головы
        logits = self.classifier(hidden_states)  # [batch_size, seq_len, 2]

        return logits
