import streamlit as st
import numpy as np
import cv2
import os
from segmentation_models import Unet
import tensorflow as tf

# Important for loading a custom-trained model
custom_objects = Unet().custom_objects
# Set framework to match training environment
os.environ["SM_FRAMEWORK"] = "tf.keras"
sm.set_framework('tf.keras')
model = tf.keras.models.load_model("my_model.h5", custom_objects=custom_objects, compile=False)




# ✅ Make sure framework is correctly set
# sm.set_framework('tf.keras')
# sm.framework()

# custom_objects = {
#     'iou_score': sm.metrics.IOUScore(threshold=0.5),
#     'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy()
# }

# ✅ Load model
# model = tf.keras.models.load_model("my_model.h5", custom_objects=custom_objects, compile=False)

# ✅ Constants
H, W = 480, 480
CLASS_COLORS = np.array([
    [0, 0, 0],      # Background
    [255, 0, 0],    # Class 1
    [0, 255, 0],    # Class 2
    [0, 0, 255]     # Class 3
], dtype=np.uint8)

# ✅ Image preprocessing
def preprocess_image(image):
    image = cv2.resize(image, (W, H))
    image = image / 255.0
    return image.astype(np.float32)

# ✅ Prediction
def predict_mask(image):
    input_img = preprocess_image(image)
    input_img = np.expand_dims(input_img, axis=0)
    pred = model.predict(input_img)[0]
    pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)
    return CLASS_COLORS[pred_mask]

# ✅ Streamlit app
st.title("Lunar Rock Segmentation")
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="Input Image", use_column_width=True)

    color_mask = predict_mask(img)
    st.image(color_mask, caption="Predicted Segmentation Mask", use_column_width=True)
