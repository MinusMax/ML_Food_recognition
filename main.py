import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Set the page configuration
st.set_page_config(
    page_title="Thai Food Recognition",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.h5")

# Load the model
model = load_model()

# TensorFlow Model Prediction
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    input_arr = input_arr / 255.0  # Normalize the image
    predictions = model.predict(input_arr)
    return predictions  # Return predictions array

# Custom CSS for styling
def apply_custom_css():
    st.markdown(""" 
        <style>
        /* Sidebar Styling */
        .css-1d391kg {
            background-color: #2e2e2e;
            color: #ffffff;
        }
        .css-1d391kg img {
            filter: brightness(70%);
        }
        .css-1d391kg .stButton>button {
            background-color: #444;
            color: #fff;
        }
        .css-1d391kg .stText {
            color: #ddd;
        }

        /* Main page background */
        .css-18e3th9 {
            background: linear-gradient(135deg, #ffb347, #ffcc33);
            color: #fff;
        }

        /* Prediction page background */
        .css-1v3fvcr {
            background: linear-gradient(135deg, #f6d365, #fda085);
        }

        /* Additional Styling */
        .css-1v3fvcr h1 {
            color: #fff;
        }
        </style>
    """, unsafe_allow_html=True)

# Apply Custom CSS
apply_custom_css()

# Sidebar
with st.sidebar:
    st.image("Thaifood_banner.jpg", use_column_width=True)  # Logo image
    st.title("Thai Food Dashboard")
    st.write("Select a page to navigate:")
    app_mode = st.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.title("Welcome to Thai Food Recognition!")
    st.markdown("""
        Discover and recognize various Thai dishes with our state-of-the-art model.
        Use the navigation menu to explore different features of this app.
    """)
    st.image("Thaifood_banner.jpg", use_column_width=True)
    st.write("Enjoy exploring the flavors of Thailand through this interactive app!")

    # Add interactive elements
    st.slider("Adjust your excitement level", 0, 100, 50)
    st.selectbox("Choose your favorite Thai dish", ["Pad Thai", "Tom Yum Goong", "Green Curry"])

# About Project
elif app_mode == "About Project":
    st.title("About the Project")
    st.markdown("""
        <h3 style='color: #ff6347;'>Dataset Details</h3>
        <p>The dataset used for this project contains images of Thai food items organized into three folders:</p>
        <ul>
            <li><strong>train</strong>: 100 images per class</li>
            <li><strong>test</strong>: 10 images per class</li>
            <li><strong>validation</strong>: 10 images per class</li>
        </ul>
    """, unsafe_allow_html=True)

# Prediction Page
elif app_mode == "Prediction":
    st.title("Model Prediction")
    st.write("Upload a Thai food image to get a prediction from the model.")
    
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict"):
            with st.spinner("Processing..."):
                # Convert the uploaded file to an image object
                image = Image.open(uploaded_file)
                
                # Make prediction
                predictions = model_prediction(uploaded_file)
                
                # Reading Labels
                try:
                    with open("labels.txt") as f:
                        labels = [line.strip() for line in f]
                except FileNotFoundError:
                    st.error("Error: `labels.txt` file not found.")
                    labels = ["Unknown"]  # Default label in case of error
                
                # Get the index of the highest prediction
                result_index = np.argmax(predictions)
                st.success(f"The model predicts this is: **{labels[result_index]}**")
                
                # Confidence Bar Chart
                st.write("Confidence Levels:")
                
                # Create a DataFrame for plotting
                confidence_df = pd.DataFrame({
                    'Label': labels,
                    'Confidence': predictions[0] * 100
                })
                
                st.bar_chart(confidence_df.set_index('Label'))
