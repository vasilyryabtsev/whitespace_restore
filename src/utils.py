import torch
import torch.nn.functional as F
import numpy as np
import importlib
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


def _make_allowlist_from_checkpoint(filepath: str):
    """
    Return a best-effort list of objects (or (obj, 'module.name') tuples)
    that should be allowlisted for a weights_only load of `filepath`.
    """
    allowlist = []
    try:
        # returns list[str] like "numpy.core.multiarray.scalar"
        unsafe_names = torch.serialization.get_unsafe_globals_in_checkpoint(filepath)
    except Exception:
        unsafe_names = []

    for fullname in unsafe_names:
        # fullname is e.g. "numpy.core.multiarray.scalar"
        try:
            module_name, _, attr = fullname.rpartition('.')
            module = importlib.import_module(module_name)
            obj = getattr(module, attr)
            # use tuple (obj, fullname) to ensure correct mapping from name -> object
            allowlist.append((obj, fullname))
        except Exception:
            # best-effort: try to import module and add module object
            try:
                mod = importlib.import_module(fullname)
                allowlist.append(mod)
            except Exception:
                # ignore anything we can't resolve
                pass

    # common numpy artifacts that often appear in checkpoints
    try:
        allowlist.append((np.core.multiarray.scalar, "numpy.core.multiarray.scalar"))
    except Exception:
        pass
    try:
        allowlist.append((np.dtype, "numpy.dtype"))
    except Exception:
        pass

    return allowlist

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads checkpoint into model (and optionally optimizer/scheduler).
    Prefers safe `weights_only=True` loads and allowlists identified globals.
    If that fails and you explicitly trust the file, it can fall back to
    weights_only=False (pickle-mode).
    """
    allowlist = _make_allowlist_from_checkpoint(filepath)
    checkpoint = None

    # If we have things to allowlist, use the safe_globals context manager
    try:
        if allowlist:
            # safe_globals accepts a list of objects **or** (obj, "module.name") tuples
            with torch.serialization.safe_globals(allowlist):
                checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        else:
            checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    except Exception as e_safe:
        # Helpful diagnostic printout for debugging
        print("Safe weights_only load failed:", e_safe)
        print("Unsafe globals (static scan):",
              torch.serialization.get_unsafe_globals_in_checkpoint(filepath))
        # Fallback: only if you trust the checkpoint, load with weights_only=False
        print("Falling back to torch.load(..., weights_only=False). "
              "WARNING: this uses pickle and can execute arbitrary code. "
              "Only proceed if you trust the checkpoint source.")
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Normal state-dict restore
    print("Loading state dict...")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Checkpoint loaded from {filepath}")
    return checkpoint

