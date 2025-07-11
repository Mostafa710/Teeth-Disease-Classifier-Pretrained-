import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model once using Streamlit cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fine_tuned_InceptionV3_classifier.h5")

# App Title
st.title("Teeth Disease Classifier 🦷")

# Class Names (Update if your model has different order)
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image of a tooth", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Preprocess image
    img_resized = image.resize((224, 224))  # Match Model's input shape
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = 100 * np.max(prediction)

    # Show result
    st.markdown(f"### 🔍 Prediction: `{pred_class}`")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
