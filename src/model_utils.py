import json
import pandas as pd
import tensorflow as tf

from src.config import MODEL_PATH, CLASS_INDICES_PATH, CALORIES_CSV_PATH


def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def save_class_indices(class_indices):
    CLASS_INDICES_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_indices, file, indent=4)


def load_class_labels():
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as file:
        class_indices = json.load(file)

    return {index: label for label, index in class_indices.items()}


def load_calorie_data():
    calories_df = pd.read_csv(CALORIES_CSV_PATH)

    return dict(
        zip(
            calories_df["food_name"],
            calories_df["calories_per_serving"]
        )
    )


def normalize_food_name(food_name):
    return (
        food_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )