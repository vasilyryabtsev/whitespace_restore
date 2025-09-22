import os
import torch
import pandas as pd
import requests
import subprocess
import tempfile

from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from .dataset import WhitespaceDataset
from .utils import load_checkpoint
from .model import ByT5WhitespaceRestorer


def restore_whitespace_batch(
    dataset, model, device="cpu", batch_size=16, num_workers=-1, threshold=0.5
):
    """Восстанавливает пробелы в текстах из датасета батчами."""
    model.eval()
    model.to(device)

    # Получаем токенизатор из датасета или создаем новый
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small", use_fast=True)

    # Создаем DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Restoring whitespace")
        ):
            # Получаем входные данные
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Получаем предсказания модели
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )  # [batch_size, seq_len, 2]
            # Получаем вероятности применив softmax
            probabilities = torch.softmax(logits, dim=-1)  # [batch_size, seq_len, 2]
            # Используем порог вероятности для класса 1 (пробел)
            space_probs = probabilities[
                :, :, 1
            ]  # [batch_size, seq_len] - вероятности пробела
            predictions = (
                space_probs > threshold
            ).long()  # [batch_size, seq_len] - 1 если > threshold

            # Обрабатываем каждый пример в батче
            for i in range(input_ids.size(0)):
                sample_idx = batch_idx * batch_size + i

                # Проверяем что индекс не выходит за границы датасета
                if sample_idx >= len(dataset):
                    break

                # Получаем исходный текст из датасета
                original_text, _ = dataset.samples[sample_idx]

                # Получаем предсказания для этого примера
                sample_input_ids = input_ids[i]
                sample_predictions = predictions[i]
                sample_attention_mask = attention_mask[i]

                # Восстанавливаем текст с пробелами
                restored_text, space_indices = restore_text_from_predictions(
                    original_text,
                    sample_input_ids,
                    sample_predictions,
                    sample_attention_mask,
                    tokenizer,
                )

                results.append(
                    {
                        "original_text": original_text,
                        "restored_text": restored_text,
                        "space_indices": space_indices,
                    }
                )

    return pd.DataFrame(results)


def restore_text_from_predictions(
    original_text, input_ids, predictions, attention_mask, tokenizer
):
    """Восстанавливает текст с пробелами на основе предсказаний модели."""

    # Получаем специальные токены
    special_tokens = get_special_token_ids(tokenizer)

    # Собираем предсказания только для валидных токенов (не специальных)
    valid_predictions = []
    token_idx = 0

    for token_id, pred, mask in zip(input_ids, predictions, attention_mask):
        # Пропускаем паддинг и специальные токены
        if mask == 0 or token_id.item() in special_tokens:
            continue

        valid_predictions.append(pred.item())
        token_idx += 1

    # Конвертируем байтовые предсказания в символьные
    char_predictions = byte_predictions_to_char_predictions(
        original_text, valid_predictions
    )

    # Строим восстановленный текст и список индексов
    restored_text = ""
    space_indices = []

    for i, char in enumerate(original_text):
        restored_text += char

        # Если предсказание говорит добавить пробел после этого символа
        if i < len(char_predictions) and char_predictions[i] == 1:
            restored_text += " "
            space_indices.append(i)

    return restored_text, space_indices


def byte_predictions_to_char_predictions(text, byte_predictions):
    """Конвертирует предсказания для байтов в предсказания для символов."""
    char_predictions = []
    byte_idx = 0

    for char in text:
        char_bytes = char.encode("utf-8")
        num_bytes = len(char_bytes)

        # Берем предсказание для последнего байта символа
        if byte_idx + num_bytes - 1 < len(byte_predictions):
            char_prediction = byte_predictions[byte_idx + num_bytes - 1]
        else:
            char_prediction = 0  # По умолчанию не добавляем пробел

        char_predictions.append(char_prediction)
        byte_idx += num_bytes

    return char_predictions


def get_special_token_ids(tokenizer):
    """Возвращает множество ID специальных токенов ByT5."""
    special_ids = {0, 1, 2}  # PAD, EOS, UNK

    # Добавляем все токены из tokenizer
    if hasattr(tokenizer, "all_special_ids"):
        special_ids.update(tokenizer.all_special_ids)

    # Extra ID токены для ByT5
    special_ids.update(range(32099, 32200))  # <extra_id_0> to <extra_id_99>

    return special_ids


def clean_txt_file(file_path):
    """
    Открывает файл, удаляет первую строчку,
    из остальных строк удаляет все символы до первой запятой включительно.
    """
    # Читаем все строки из файла
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Пропускаем первую строку и обрабатываем остальные
    cleaned_lines = []
    for line in lines[1:]:
        # Находим первую запятую и удаляем все до неё включительно
        comma_index = line.find(',')
        cleaned_line = line[comma_index + 1:]  # Берём всё после запятой
        cleaned_lines.append(cleaned_line)
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8')

    # Записываем обработанные строки в новый файл
    temp_file.writelines(cleaned_lines)
    temp_file.close()
    
    return temp_file.name


def inference(
    path_to_data,
    path_to_save,
    path_to_weights,
    device="cpu",
    batch_size=4,
    num_workers=2,
    threshold=0.5,
):
    """Выполняет инференс модели на данных и сохраняет результаты."""
    # Очистка данных
    temp_path = clean_txt_file(path_to_data)
    # Создание датасета
    dataset = WhitespaceDataset(temp_path)
    print(f"Dataset created with {len(dataset)} samples")
    # Загрузка модели
    print("Initializing model...")
    model = ByT5WhitespaceRestorer()
    print(f"Loading model weights from {path_to_weights}...")
    load_checkpoint(path_to_weights, model, device=device)
    model.to(device)
    model.eval()
    # Получение предсказаний
    print("Starting inference...")
    preds = restore_whitespace_batch(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        threshold=threshold,
    )
    # Сохранение результатов
    print(f"Saving results to {path_to_save}")
    res = pd.DataFrame({"id": preds.index, "predicted_positions": preds.space_indices})
    res.to_csv(path_to_save, index=False)
    # Удаление временного файла
    os.remove(temp_path)


def download_and_extract_file(filename, download_url):
    """
    Проверяет наличие файла в текущей директории. Если файл не найден,
    скачивает архив с Google Drive, извлекает его и возвращает путь к файлу.
    """
    current_dir = os.getcwd()
    
    # Проверяем есть ли файл в текущей директории
    file_path = os.path.join(current_dir, filename)
    if os.path.exists(file_path):
        return file_path
    
    zip_filename = "weights.zip"

    # Скачиваем файл с помощью wget
    subprocess.run([
        "wget", 
        "--no-check-certificate", 
        download_url, 
        "-O", 
        zip_filename
    ], check=True)
    
    # Извлекаем архив
    subprocess.run(["unzip", zip_filename], check=True)
    
    # Удаляем архив
    os.remove(zip_filename)
    
    # Возвращаем путь к файлу
    return os.path.join(current_dir, filename)
