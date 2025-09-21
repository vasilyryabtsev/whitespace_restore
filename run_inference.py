import os
import logging

from dotenv import load_dotenv
from src.api import inference, download_weights


def main():
    """Основной скрипт для запуска инференса модели восстановления пробелов."""
    load_dotenv()

    weights_url = os.getenv("WEIGHTS_URL", "dlskflsdkjfl")
    path_to_weights = os.getenv("PATH_TO_WEIGHTS", "./best_model.pth")
    path_to_data = os.getenv("PATH_TO_DATA")
    assert path_to_data is not None, "PATH_TO_DATA must be set in .env"
    path_to_save = os.getenv("PATH_TO_SAVE")
    assert path_to_save is not None, "PATH_TO_SAVE must be set in .env"
    text_column = os.getenv("TEXT_COLUMN")
    assert text_column is not None, "TEXT_COLUMN must be set in .env"
    device = os.getenv("DEVICE")
    assert device is not None, "DEVICE must be set in .env (e.g., 'cpu' or 'cuda')"
    batch_size = int(os.getenv("BATCH_SIZE", 16))
    num_workers = int(os.getenv("NUM_WORKERS", 1))
    threshold = float(os.getenv("THRESHOLD", 0.5))

    # Загрузка весов
    logging.info("Downloading model weights...")
    download_weights(url=weights_url, path=path_to_weights)
    # Инференс
    logging.info("Running inference...")
    inference(
        path_to_data=path_to_data,
        path_to_save=path_to_save,
        path_to_weights=path_to_weights,
        text_col=text_column,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        threshold=threshold,
    )
    logging.info("Inference completed!")


if __name__ == "__main__":
    main()
