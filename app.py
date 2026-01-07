import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

model = tf.keras.models.load_model("model/disease_model.h5")

with open("knowledge_base/disease_info.json") as f:
    disease_info = json.load(f)

st.title("🌱 AI Crop Disease & Pest Scout")

uploaded_file = st.file_uploader("Upload crop leaf image", type=["jpg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((128,128))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    disease = list(disease_info.keys())[class_index]

    st.subheader(f"Detected Disease: {disease}")
    st.write("Cause:", disease_info[disease]["cause"])
    st.write("Treatment:", disease_info[disease]["treatment"])
    st.write("Prevention:", disease_info[disease]["prevention"])
