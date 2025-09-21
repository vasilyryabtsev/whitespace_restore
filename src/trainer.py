import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def focal_loss(logits, labels, alpha=1.0, gamma=2.0, ignore_index=-100):
    """
    Focal Loss для работы с несбалансированными классами.

    Args:
        logits: Предсказания модели [batch_size * seq_len, num_classes]
        labels: Истинные метки [batch_size * seq_len]
        alpha: Вес для балансировки классов
        gamma: Фокусирующий параметр
        ignore_index: Индекс для игнорирования

    Returns:
        torch.Tensor: Focal loss
    """
    ce_loss = F.cross_entropy(
        logits, labels, reduction="none", ignore_index=ignore_index
    )
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    # Игнорируем значения с ignore_index
    mask = labels != ignore_index
    return focal_loss[mask].mean()


def calculate_metrics(logits, labels, ignore_index=-100):
    """
    Вычисляет accuracy, precision, recall, F1 для валидных токенов.

    Args:
        logits: Предсказания модели [batch_size * seq_len, 2]
        labels: Истинные метки [batch_size * seq_len]
        ignore_index: Индекс для игнорирования при вычислении метрик

    Returns:
        dict: Словарь с метриками
    """
    # Фильтруем ignore_index
    mask = labels != ignore_index

    preds = torch.argmax(logits, dim=-1)
    valid_preds = preds[mask].cpu().numpy()
    valid_labels = labels[mask].cpu().numpy()

    acc = accuracy_score(valid_labels, valid_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_preds, average="binary", zero_division=0
    )

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def train_epoch(
    model, dataloader, optimizer, scheduler=None, device="cuda", max_grad_norm=1.0
):
    """
    Обучение модели на одной эпохе с вычислением метрик.

    Args:
        model: Модель для обучения
        dataloader: DataLoader с тренировочными данными
        optimizer: Оптимизатор
        scheduler: Learning rate scheduler (опционально)
        device: Устройство для вычислений
        max_grad_norm: Максимальная норма градиентов для clipping

    Returns:
        tuple: (средний loss, словарь с метриками)
    """
    model.train()
    total_loss = 0
    all_logits = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

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

        # Собираем предсказания для метрик
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # Вычисляем метрики
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = calculate_metrics(all_logits.view(-1, 2), all_labels.view(-1))

    avg_loss = total_loss / len(dataloader)
    print(f"Train - Loss: {avg_loss:.4f}, F1: {metrics.get('f1', 0):.4f}")

    return avg_loss, metrics


def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, filepath):
    """
    Сохраняет полный чекпоинт модели с состоянием оптимизатора и scheduler.

    Args:
        model: Модель для сохранения
        optimizer: Оптимизатор
        scheduler: Learning rate scheduler (может быть None)
        epoch: Номер эпохи
        loss: Значение loss
        metrics: Словарь с метриками
        filepath: Путь для сохранения чекпоинта
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def validate_epoch(model, dataloader, device="cuda"):
    """
    Валидация модели на одной эпохе с вычислением метрик.
    Оптимизировано для Google Colab: метрики вычисляются на CPU для экономии GPU памяти.

    Args:
        model: Модель для валидации
        dataloader: DataLoader с валидационными данными
        device: Устройство для вычислений

    Returns:
        tuple: (средний loss, словарь с метриками)
    """
    model.eval()
    total_loss = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Получаем логиты от модели
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Вычисляем loss с учетом ignore_index=-100
            loss = focal_loss(
                logits.view(-1, 2),  # [batch_size * seq_len, 2]
                labels.view(-1),  # [batch_size * seq_len]
                gamma=2.0,
            )

            total_loss += loss.item()

            # Собираем предсказания для метрик
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Вычисляем метрики
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = calculate_metrics(all_logits.view(-1, 2), all_labels.view(-1))

    avg_loss = total_loss / len(dataloader)
    print(f"Val - Loss: {avg_loss:.4f}, F1: {metrics.get('f1', 0):.4f}")

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
    """
    Обучение и валидация модели с сохранением чекпоинтов.

    Args:
        model: Модель для обучения
        train_loader: DataLoader с тренировочными данными
        val_loader: DataLoader с валидационными данными
        optimizer: Оптимизатор
        scheduler: Learning rate scheduler (опционально)
        device: Устройство для вычислений
        epochs: Количество эпох
        save_dir: Директория для сохранения чекпоинтов

    Returns:
        dict: История обучения с метриками
    """
    # Создаем директорию для чекпоинтов
    os.makedirs(save_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []}

    print(f"Starting training for {epochs} epochs")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Обучение
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )

        # Валидация
        val_loss, val_metrics = validate_epoch(model, val_loader, device)

        # Сохраняем историю
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)

        # Логирование результатов
        print(f'Train - Loss: {train_loss:.4f}, F1: {train_metrics.get("f1", 0):.4f}')
        print(f'Val - Loss: {val_loss:.4f}, F1: {val_metrics.get("f1", 0):.4f}')

        # Сохраняем последний чекпоинт
        last_checkpoint_path = os.path.join(save_dir, "last_checkpoint.pt")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            val_loss,
            val_metrics,
            last_checkpoint_path,
        )

    return history
