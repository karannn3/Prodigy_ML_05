import tempfile
from pathlib import Path

import streamlit as st

from src.predict import predict_food_and_calories


st.set_page_config(
    page_title="Food Calories Detection",
    page_icon="🍎",
    layout="centered",
)

st.title("Food Calories Detection")
st.write("Upload a food image to predict the food type and estimated calories.")

uploaded_file = st.file_uploader(
    "Upload food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_image_path = temp_file.name

    with st.spinner("Predicting..."):
        result = predict_food_and_calories(temp_image_path)

    st.subheader("Prediction Result")

    st.write(f"**Food:** {result['food']}")
    st.write(f"**Confidence:** {result['confidence']}%")
    st.write(f"**Estimated Calories:** {result['calories']} kcal per serving")