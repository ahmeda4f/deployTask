import streamlit as st
import numpy as np
from PIL import Image
import keras
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.resnet import preprocess_input

st.title("🧠 Brain Tumour Project")
task = st.radio("Choose a task", ["Classification", "Segmentation"])
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model(task):
    if task == "Classification":
        model_path = hf_hub_download(
            repo_id="Ahmed-Ashraf-00/brain_tumour_testing",
            filename="best_model_classification.keras",
            token=st.secrets.get("HF_TOKEN")
        )
    else:
        model_path = hf_hub_download(
            repo_id="Ahmed-Ashraf-00/brain_tumour_testing",
            filename="best_model_segmentation_1.keras",
            token=st.secrets.get("HF_TOKEN")
        )
    return keras.models.load_model(model_path, compile=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    model = load_model(task)

    if task == "Classification":
        img = image.resize((224, 224))
        arr = np.array(img)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = preprocess_input(arr.astype(np.float32))
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr)[0][0]
        label = "no" if preds > 0.5 else "yes"
        st.write("Prediction:", label)
    else:
        img = image.resize((128, 128))
        arr = np.array(img) / 255.0
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = np.expand_dims(arr, axis=0)
        mask = model.predict(arr)[0]
        mask = (mask > 0.5).astype(np.uint8) * 255
        st.image(mask, caption="Predicted Mask", use_container_width=True)



