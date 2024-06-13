import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Define the class names
class_names = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'radish', 'soy beans', 'spinach', 'sweetcorn',
    'tomato', 'turnip', 'watermelon'
]





# Load the trained model
model = tf.keras.models.load_model('fruit_vegetable_classifier.h5')

def load_and_prep_image(img, img_size=224):
    """
    Reads an image from disk, turns it into a tensor, and reshapes it to (img_size, img_size, 3).
    """
    img = img.resize((img_size, img_size))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def predict_image(model, img):
    """
    Predicts the class of an image using a trained model.
    """
    img = load_and_prep_image(img)
    pred = model.predict(img)
    pred_class = class_names[np.argmax(pred)]
    return pred_class

# Streamlit UI
st.title("Fruit and Vegetable Image Classifier")
st.write("Upload an image of a fruit or vegetable and the classifier will predict its category.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict_image(model, image)
    st.write(f"The image is classified as: **{prediction}**")
