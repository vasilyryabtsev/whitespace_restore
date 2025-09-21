import torch
import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


def restore_whitespace_batch(dataset, model, device='cuda', batch_size=16, num_workers=-1, threshold=0.5):
    """
    Восстанавливает пробелы в текстах из датасета используя обученную модель.
    
    Args:
        dataset: WhitespaceDataset с текстами для восстановления
        model: Обученная модель ByT5WhitespaceRestorer
        device: Устройство для вычислений ('cuda' или 'cpu')
        batch_size: Размер батча для обработки
        num_workers: Количество процессов для загрузки данных
        threshold: Порог вероятности для добавления пробела (0.0-1.0)
        
    Returns:
        pd.DataFrame: DataFrame со столбцами:
            - 'original_text': исходный текст без пробелов
            - 'restored_text': текст с восстановленными пробелами
            - 'space_indices': список индексов символов после которых нужен пробел
    """
    model.eval()
    model.to(device)
    
    # Получаем токенизатор из датасета или создаем новый
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small", use_fast=True)
    
    # Создаем DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Restoring whitespace")):
            # Получаем входные данные
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Получаем предсказания модели
            logits = model(input_ids=input_ids, attention_mask=attention_mask)  # [batch_size, seq_len, 2]
            # Получаем вероятности применив softmax
            probabilities = torch.softmax(logits, dim=-1)  # [batch_size, seq_len, 2]
            # Используем порог вероятности для класса 1 (пробел)
            space_probs = probabilities[:, :, 1]  # [batch_size, seq_len] - вероятности пробела
            predictions = (space_probs > threshold).long()  # [batch_size, seq_len] - 1 если > threshold
            
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
                    tokenizer
                )
                
                results.append({
                    'original_text': original_text,
                    'restored_text': restored_text,
                    'space_indices': space_indices
                })
    
    return pd.DataFrame(results)


def restore_text_from_predictions(original_text, input_ids, predictions, attention_mask, tokenizer):
    """
    Восстанавливает текст с пробелами на основе предсказаний модели.
    
    Args:
        original_text: Исходный текст без пробелов
        input_ids: Токены входной последовательности
        predictions: Предсказания модели (0 или 1 для каждого токена)
        attention_mask: Маска внимания
        tokenizer: Токенизатор ByT5
        
    Returns:
        tuple: (восстановленный_текст, список_индексов_пробелов)
    """
    
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
    char_predictions = byte_predictions_to_char_predictions(original_text, valid_predictions)
    
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
    """
    Конвертирует предсказания для байтов в предсказания для символов.
    Предсказание для символа берется из последнего байта этого символа.
    
    Args:
        text: Исходный текст
        byte_predictions: Список предсказаний для каждого байта
        
    Returns:
        list: Предсказания для каждого символа
    """
    char_predictions = []
    byte_idx = 0
    
    for char in text:
        char_bytes = char.encode('utf-8')
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
    """Возвращает множество ID всех специальных токенов ByT5"""
    special_ids = {0, 1, 2}  # PAD, EOS, UNK
    
    # Добавляем все токены из tokenizer
    if hasattr(tokenizer, "all_special_ids"):
        special_ids.update(tokenizer.all_special_ids)
    
    # Extra ID токены для ByT5
    special_ids.update(range(32099, 32200))  # <extra_id_0> to <extra_id_99>
    
    return special_ids
