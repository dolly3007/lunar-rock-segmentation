import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import segmentation_models as sm

# Define custom objects
custom_objects = {
    'iou_score': sm.metrics.IOUScore(threshold=0.5),
    'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy()
}

# Load model with custom objects
model = tf.keras.models.load_model("my_model.h5", custom_objects=custom_objects, compile=False)

H, W = 480, 480
CLASS_COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255]
], dtype=np.uint8)

def preprocess_image(image):
    image = cv2.resize(image, (W, H))
    image = image / 255.0
    return image.astype(np.float32)

def predict_mask(image):
    input_img = preprocess_image(image)
    input_img = np.expand_dims(input_img, axis=0)
    pred = model.predict(input_img)[0]
    pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)
    return CLASS_COLORS[pred_mask]

# Streamlit app UI
st.title("Lunar_Rock Segmentation")
uploaded = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="Input Image", use_column_width=True)

    color_mask = predict_mask(img)
    st.image(color_mask, caption="Predicted Segmentation Mask", use_column_width=True)
