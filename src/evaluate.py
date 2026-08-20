from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import IMG_SIZE, EVALUATION_DIR
from src.model_utils import load_model


BATCH_SIZE = 16


def evaluate_model():
    model = load_model()

    evaluation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    evaluation_data = evaluation_datagen.flow_from_directory(
        EVALUATION_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    loss, accuracy = model.evaluate(evaluation_data)

    print(f"Evaluation Loss: {loss:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    evaluate_model()