import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class WhitespaceDataset(Dataset):
    """
    Датасет для задачи восстановления пробелов (двухклассовая постановка: K=0, I=1).
    Используется байтовый токенизатор ByT5.
    Вход: строка без пробелов, таргет: строка с правильными пробелами.
    """

    def __init__(self, filepath, max_length=128, ignore_index=-100):
        self.samples = []
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/byt5-small", use_fast=True
        )
        self.max_length = max_length
        self.ignore_index = ignore_index

        with open(filepath, encoding="utf-8") as file:
            for line in file:
                target_text = " ".join(line.split())  # Удаляем лишние пробелы
                if not target_text:
                    continue
                input_text = target_text.replace(" ", "")  # Убираем пробелы
                self.samples.append((input_text, target_text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_text, target_text = self.samples[idx]

        # Генерируем метки по символам
        char_labels = self._make_labels(input_text, target_text)

        # Токенизация входа
        enc = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)

        # Выравнивание меток с токенами через UTF-8 байты
        labels = self._align_labels_with_byt5_tokens(char_labels, input_text, input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _get_special_token_ids(self):
        """Возвращает множество ID всех специальных токенов ByT5"""
        special_ids = {0, 1, 2}  # PAD, EOS, UNK

        # Добавляем все токены из tokenizer
        if hasattr(self.tokenizer, "all_special_ids"):
            special_ids.update(self.tokenizer.all_special_ids)

        # Extra ID токены для ByT5
        special_ids.update(range(32099, 32200))  # <extra_id_0> to <extra_id_99>

        return special_ids

    def _align_labels_with_byt5_tokens(self, char_labels, input_text, token_ids):
        """
        Выравнивает символьные метки с байтовыми токенами ByT5.
        ByT5 токенизирует текст в UTF-8 байты, где каждый токен = байт + 3.
        """
        # Конвертируем символьные метки в байтовые
        byte_labels = self._char_labels_to_byte_labels(input_text, char_labels)

        # Получаем все специальные токены
        special_tokens = self._get_special_token_ids()

        labels = []
        byte_idx = 0

        for token_id in token_ids:
            # Проверяем специальные токены
            if token_id.item() in special_tokens:
                labels.append(self.ignore_index)
            else:
                # Обычный байт-токен (байт + 3)
                if byte_idx < len(byte_labels):
                    labels.append(byte_labels[byte_idx])
                    byte_idx += 1
                else:
                    # Паддинг или обрезка
                    labels.append(self.ignore_index)

        return labels

    def _char_labels_to_byte_labels(self, input_text, char_labels):
        """
        Конвертирует метки символов в метки UTF-8 байтов.
        Метка символа присваивается последнему байту этого символа.
        """
        byte_labels = []

        for i, char in enumerate(input_text):
            char_bytes = char.encode("utf-8")
            # Чтобы не выйти за границы char_labels
            label = char_labels[i] if i < len(char_labels) else 0

            # Все байты символа получают метку 0, кроме последнего
            byte_labels.extend([0] * (len(char_bytes) - 1))
            byte_labels.append(label)

        return byte_labels

    @staticmethod
    def _make_labels(input_text, target_text):
        """
        Строит список меток длиной = числу символов input_text:
        1 (I) - после символа нужно вставить пробел
        0 (K) - пробел не нужен
        """
        labels = []
        target_pos = 0

        for char in input_text:
            while target_pos < len(target_text) and target_text[target_pos] == " ":
                target_pos += 1

            if target_pos < len(target_text) and target_text[target_pos] == char:
                target_pos += 1
                if target_pos < len(target_text) and target_text[target_pos] == " ":
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(0)

        return labels
