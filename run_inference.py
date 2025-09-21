import os
from dotenv import load_dotenv
from src.api import inference, download_weights


load_dotenv()

WEIGHTS_URL = os.getenv("WEIGHTS_URL", "dlskflsdkjfl")
PATH_TO_WEIGHTS = os.getenv("PATH_TO_WEIGHTS", "./best_model.pth")
PATH_TO_DATA = os.getenv("PATH_TO_DATA")
assert PATH_TO_DATA is not None, "PATH_TO_DATA must be set in .env"
PATH_TO_SAVE = os.getenv("PATH_TO_SAVE")
assert PATH_TO_SAVE is not None, "PATH_TO_SAVE must be set in .env"
TEXT_COLUMN = os.getenv("TEXT_COLUMN")
assert TEXT_COLUMN is not None, "TEXT_COLUMN must be set in .env"
DEVICE = os.getenv("DEVICE")
assert DEVICE is not None, "DEVICE must be set in .env (e.g., 'cpu' or 'cuda')"

if __name__ == "__main__":
    # Загрузка весов
    logging.info("Downloading model weights...")
    download_weights(url=WEIGHTS_URL, path=PATH_TO_WEIGHTS)
    # Инференс
    logging.info("Running inference...")
    inference(
        path_to_data=PATH_TO_DATA,
        path_to_save=PATH_TO_SAVE,
        path_to_weights=PATH_TO_WEIGHTS,
        text_col=TEXT_COLUMN,
        device=DEVICE,
        batch_size=16,
        num_workers=4,
        threshold=0.5
    )
    logging.info("Inference completed!")
