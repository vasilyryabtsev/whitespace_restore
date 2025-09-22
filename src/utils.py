import torch
import torch.nn.functional as F
import os

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def focal_loss(logits, labels, alpha=1.0, gamma=2.0, ignore_index=-100):
    """Вычисляет Focal Loss для несбалансированных классов."""
    ce_loss = F.cross_entropy(
        logits, labels, reduction="none", ignore_index=ignore_index
    )
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    # Игнорируем значения с ignore_index
    mask = labels != ignore_index
    return focal_loss[mask].mean()


def calculate_metrics(logits, labels, ignore_index=-100):
    """Вычисляет accuracy, precision, recall и F1 для валидных токенов."""
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


def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, filepath):
    """Сохраняет чекпоинт модели с состоянием оптимизатора и метриками."""
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


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """Загружает чекпоинт модели и восстанавливает состояние."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    print("Loading state dict...")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Checkpoint loaded from {filepath}")
