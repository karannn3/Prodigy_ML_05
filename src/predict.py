import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import IMG_SIZE
from src.model_utils import (
    load_model,
    load_class_labels,
    load_calorie_data,
    normalize_food_name,
)


model = load_model()
labels = load_class_labels()
calorie_data = load_calorie_data()


def predict_food_and_calories(image_path):
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)

    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    predicted_class = labels[class_index]
    normalized_class = normalize_food_name(predicted_class)

    calories = calorie_data.get(normalized_class, "Not Available")

    return {
        "food": predicted_class,
        "confidence": round(confidence, 2),
        "calories": calories,
    }