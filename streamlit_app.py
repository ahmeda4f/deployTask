import streamlit as st
import numpy as np
from PIL import Image
import keras

st.title("🧠 Brain Tumour Project")
task = st.radio("Choose a task", ["Classification", "Segmentation"])
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model(task):
    if task == "Classification":
        model_path = hf_hub_download(
            repo_id="Ahmed-Ashraf-00/brain_tumour_testing",
            filename="content/best_model_classification.keras",
            token=st.secrets.get("HF_TOKEN")
        )
    else:
        model_path = hf_hub_download(
            repo_id="Ahmed-Ashraf-00/brain_tumour_testing",
            filename="content/best_model_segmentation.keras",
            token=st.secrets.get("HF_TOKEN")
        )
    return keras.models.load_model(model_path, compile=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    model = load_model(task)

    if task == "Classification":
        img = image.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr)
        label = np.argmax(preds, axis=1)[0]
        st.write("Prediction:", label)

    else:
        img = image.resize((128, 128))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        mask = model.predict(arr)[0]
        mask = (mask > 0.5).astype(np.uint8) * 255
        st.image(mask, caption="Predicted Mask", use_container_width=True)

