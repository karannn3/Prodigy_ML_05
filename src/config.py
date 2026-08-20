from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "food11_model.keras"
CLASS_INDICES_PATH = BASE_DIR / "artifacts" / "class_indices.json"
CALORIES_CSV_PATH = BASE_DIR / "data" / "calories.csv"

IMG_SIZE = 160

TRAIN_DIR = BASE_DIR / "dataset" / "training"
VALIDATION_DIR = BASE_DIR / "dataset" / "validation"
EVALUATION_DIR = BASE_DIR / "dataset" / "evaluation"