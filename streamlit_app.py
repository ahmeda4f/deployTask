import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow import keras
import numpy as np
from PIL import Image

st.title("🧠 Brain Tumour Project")
st.write("Choose a task (Classification or Segmentation) and upload an image.")

task = st.selectbox("Choose a task", ["Classification", "Segmentation"])

model_files = {
    "Classification": "best_model_classification.keras",
    "Segmentation": "best_model_segmentation.keras"
}

@st.cache_resource
def load_model(task_name):
    model_path = hf_hub_download(
        repo_id="Ahmed-Ashraf-00/brain_tumour_testing",
        filename=model_files[task_name]
    )
    return keras.models.load_model(model_path)

model = load_model(task)

uploaded_file = st.file_uploader("Upload an MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if task == "Classification":
        pred = model.predict(img_array)
        st.write("Prediction:", pred)
    else:
        mask = model.predict(img_array)[0]
        st.image(mask, caption="Predicted Mask")
