import os
import torch

from tqdm import tqdm
from datetime import datetime
from .utils import save_checkpoint, calculate_metrics, focal_loss
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


def train_epoch(
    model, dataloader, optimizer, scheduler=None, device="cuda", max_grad_norm=1.0
):
    """Обучает модель на одной эпохе и возвращает loss с метриками."""
    model.train()
    total_loss = 0
    batch_metrics = []

    for batch in tqdm(dataloader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast():
            # Получаем логиты от модели
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Вычисляем focal loss с gamma=2
            loss = focal_loss(
                logits.view(-1, 2),  # [batch_size * seq_len, 2]
                labels.view(-1),  # [batch_size * seq_len]
                gamma=2.0,
            )

        loss.backward()

        # Gradient clipping для стабильности обучения
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        # Вычисляем метрики для текущего батча
        batch_metrics_dict = calculate_metrics(logits.view(-1, 2), labels.view(-1))
        batch_metrics.append(batch_metrics_dict)

    # Вычисляем средние метрики по всем батчам
    metrics = {}
    for key in batch_metrics[0].keys():
        metrics[key] = sum(batch[key] for batch in batch_metrics) / len(batch_metrics)

    avg_loss = total_loss / len(dataloader)

    return avg_loss, metrics


def validate_epoch(model, dataloader, device="cuda"):
    """Валидирует модель на одной эпохе и возвращает loss с метриками."""
    model.eval()
    total_loss = 0
    batch_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                # Получаем логиты от модели
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

                # Вычисляем loss с учетом ignore_index=-100
                loss = focal_loss(
                    logits.view(-1, 2),  # [batch_size * seq_len, 2]
                    labels.view(-1),  # [batch_size * seq_len]
                    gamma=2.0,
                )

            total_loss += loss.item()

            # Вычисляем метрики для текущего батча
            batch_metrics_dict = calculate_metrics(logits.view(-1, 2), labels.view(-1))
            batch_metrics.append(batch_metrics_dict)

    # Вычисляем средние метрики по всем батчам
    metrics = {}
    for key in batch_metrics[0].keys():
        metrics[key] = sum(batch[key] for batch in batch_metrics) / len(batch_metrics)

    avg_loss = total_loss / len(dataloader)

    return avg_loss, metrics


def train_and_validate(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler=None,
    device="cuda",
    epochs=3,
    save_dir="checkpoints",
):
    """Обучает и валидирует модель с сохранением чекпоинтов."""
    # Создаем директорию для чекпоинтов
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting training for {epochs} epochs")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Обучение
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        print(f"Train - Loss: {train_loss:.4f}, F1: {train_metrics.get('f1', 0):.4f}")

        # Валидация
        val_loss, val_metrics = validate_epoch(model, val_loader, device)
        print(f"Val   - Loss: {val_loss:.4f}, F1: {val_metrics.get('f1', 0):.4f}")

        # Сохраняем последний чекпоинт
        last_checkpoint_path = os.path.join(
            save_dir, f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            val_loss,
            val_metrics,
            last_checkpoint_path,
        )
