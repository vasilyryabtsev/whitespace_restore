import os

from dotenv import load_dotenv
from src.api import inference, download_and_extract_file

import torch
import logging

# ------------------------------ #
# Для корректной загрузки весов (Source: https://github.com/ltdrdata/comfyui-unsafe-torch)
orig_torch_load = torch.load

def torch_wrapper(*args, **kwargs):
    logging.warning("[comfyui-unsafe-torch] I have unsafely patched `torch.load`.  The `weights_only` option of `torch.load` is forcibly disabled.")
    kwargs['weights_only'] = False

    return orig_torch_load(*args, **kwargs)

torch.load = torch_wrapper

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']
# ------------------------------ #

def main():
    """Основной скрипт для запуска инференса модели восстановления пробелов."""
    load_dotenv()

    weights_url = os.getenv("WEIGHTS_URL", "https://drive.google.com/uc?export=download&id=1iywF9EK5jUf4dyl62RINw6HrUSOgR-R1")
    weights_filename = os.getenv("WEIGHTS_FILENAME", "checkpoint_20250922_095517.pt")
    path_to_data = os.getenv("PATH_TO_DATA")
    assert path_to_data is not None, "PATH_TO_DATA must be set in .env"
    path_to_save = os.getenv("PATH_TO_SAVE")
    assert path_to_save is not None, "PATH_TO_SAVE must be set in .env"
    device = os.getenv("DEVICE")
    assert device is not None, "DEVICE must be set in .env (e.g., 'cpu' or 'cuda')"
    batch_size = int(os.getenv("BATCH_SIZE", 16))
    num_workers = int(os.getenv("NUM_WORKERS", 1))
    threshold = float(os.getenv("THRESHOLD", 0.6))

    # Загрузка весов
    print("Downloading model weights...")
    path_to_weights = download_and_extract_file(weights_filename, weights_url)
    # Инференс
    print("Running inference...")
    inference(
        path_to_data=path_to_data,
        path_to_save=path_to_save,
        path_to_weights=path_to_weights,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        threshold=threshold,
    )
    print("Inference completed!")


if __name__ == "__main__":
    main()
