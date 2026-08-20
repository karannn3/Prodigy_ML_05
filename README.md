\# Food Calories Detection

This project predicts the type of food from an uploaded image and estimates its calorie content per serving using a deep learning image classification model.

## Features

- Food image classification
- Calorie estimation per serving
- Prediction confidence score
- Streamlit web app for deployment
- Separate training, evaluation, and prediction scripts

## Project Structure

```text
food calories dectation/
  app.py
  requirements.txt
  README.md
  src/
    config.py
    train.py
    predict.py
    evaluate.py
    model_utils.py
  artifacts/
    food11_model.keras
    class_indices.json
  data/
    calories.csv
  dataset/
    training/
    validation/
    evaluation/
  experiments/
    food_track.ipynb


##Technologies Used
- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pandas
- Pillow
- Streamlit



\## 📂 Dataset

dataset/
  training/
    Bread/
    Dairy product/
    Dessert/
    Egg/
    Fried food/
    Meat/
    Noodles-Pasta/
    Rice/
    Seafood/
    Soup/
    Vegetable-Fruit/
  validation/
    ...
  evaluation/
    ...


##Installation

pip install -r requirements.txt


##Train The Model

python -m src.train


##Evaluate The Model

python -m src.evaluate


##Run The Streamlit App

streamlit run app.py


##Then upload a food image to get:
- predicted food class
- confidence score
- estimated calories per serving


##Calorie Mapping
Calorie information is stored in:
--data/calories.csv


##Limitations
- Calories are estimated per serving, not calculated from actual food weight or portion size.
- Predictions depend on the quality and variety of the training dataset.
- The model can classify only the 11 food categories present in the dataset.