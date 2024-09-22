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

# Calorie information for each food label
calorie_info = {
    "Baked Prawns With Vermicelli": {
        "100g": "250-300 kcal",
        "250g": "625-750 kcal",
        "500g": "1250-1500 kcal",
        "1kg": "2500-3000 kcal"
    },
    "Banana in Coconut Milk": {
        "100g": "200-250 kcal",
        "250g": "500-625 kcal",
        "500g": "1000-1250 kcal",
        "1kg": "2000-2500 kcal"
    },
    "Chicken Green Curry": {
        "100g": "400-500 kcal",
        "250g": "1000-1250 kcal",
        "500g": "2000-2500 kcal",
        "1kg": "4000-5000 kcal"
    },
    "Chicken Mussaman Curry": {
        "100g": "450-550 kcal",
        "250g": "1125-1375 kcal",
        "500g": "2250-2750 kcal",
        "1kg": "4500-5500 kcal"
    },
    "Coconut Rice Pancake": {
        "100g": "250-300 kcal",
        "250g": "625-750 kcal",
        "500g": "1250-1500 kcal",
        "1kg": "2500-3000 kcal"
    },
    "Curried Noodle Soup with Chicken": {
        "100g": "350-400 kcal",
        "250g": "875-1000 kcal",
        "500g": "1750-2000 kcal",
        "1kg": "3500-4000 kcal"
    },
    "Egg and Pork in Sweet Brown Sauce": {
        "100g": "300-350 kcal",
        "250g": "750-875 kcal",
        "500g": "1500-1750 kcal",
        "1kg": "3000-3500 kcal"
    },
    "Egg with Tamarind Sauce": {
        "100g": "250-300 kcal",
        "250g": "625-750 kcal",
        "500g": "1250-1500 kcal",
        "1kg": "2500-3000 kcal"
    },
    "Fried Cabbage with Fish Sauce": {
        "100g": "150-200 kcal",
        "250g": "375-500 kcal",
        "500g": "750-1000 kcal",
        "1kg": "1500-2000 kcal"
    },
    "Fried Egg": {
        "100g": "200-250 kcal",
        "250g": "500-625 kcal",
        "500g": "1000-1250 kcal",
        "1kg": "2000-2500 kcal"
    },
    "Fried Rice": {
        "100g": "350-450 kcal",
        "250g": "875-1125 kcal",
        "500g": "1750-2250 kcal",
        "1kg": "3500-4500 kcal"
    },
    "Grilled River Prawn": {
        "100g": "200-250 kcal",
        "250g": "500-625 kcal",
        "500g": "1000-1250 kcal",
        "1kg": "2000-2500 kcal"
    },
    "Mango Sticky Rice": {
        "100g": "300-350 kcal",
        "250g": "750-875 kcal",
        "500g": "1500-1750 kcal",
        "1kg": "3000-3500 kcal"
    },
    "Omelet": {
        "100g": "200-250 kcal",
        "250g": "500-625 kcal",
        "500g": "1000-1250 kcal",
        "1kg": "2000-2500 kcal"
    },
    "Pork Chopped Tofu Soup": {
        "100g": "150-200 kcal",
        "250g": "375-500 kcal",
        "500g": "750-1000 kcal",
        "1kg": "1500-2000 kcal"
    },
    "Pork Curry With Morning Glory": {
        "100g": "300-400 kcal",
        "250g": "750-1000 kcal",
        "500g": "1500-2000 kcal",
        "1kg": "3000-4000 kcal"
    },
    "Shrimp Fried Rice": {
        "100g": "350-450 kcal",
        "250g": "875-1125 kcal",
        "500g": "1750-2250 kcal",
        "1kg": "3500-4500 kcal"
    },
    "Shrimp Paste Fried Rice": {
        "100g": "350-450 kcal",
        "250g": "875-1125 kcal",
        "500g": "1750-2250 kcal",
        "1kg": "3500-4500 kcal"
    },
    "Sour Soup": {
        "100g": "100-150 kcal",
        "250g": "250-375 kcal",
        "500g": "500-750 kcal",
        "1kg": "1000-1500 kcal"
    },
    "Spicy Mixed Vegetable Soup": {
        "100g": "100-150 kcal",
        "250g": "250-375 kcal",
        "500g": "500-750 kcal",
        "1kg": "1000-1500 kcal"
    },
    "Steamed Capon in Flavored Rice": {
        "100g": "300-350 kcal",
        "250g": "750-875 kcal",
        "500g": "1500-1750 kcal",
        "1kg": "3000-3500 kcal"
    },
    "Stir Fried Chicken with Chestnuts": {
        "100g": "350-400 kcal",
        "250g": "875-1000 kcal",
        "500g": "1750-2000 kcal",
        "1kg": "3500-4000 kcal"
    },
    "Stir Fried Rice Noodles with Chicken": {
        "100g": "400-500 kcal",
        "250g": "1000-1250 kcal",
        "500g": "2000-2500 kcal",
        "1kg": "4000-5000 kcal"
    },
    "Stuffed Bitter Gourd Broth": {
        "100g": "150-200 kcal",
        "250g": "375-500 kcal",
        "500g": "750-1000 kcal",
        "1kg": "1500-2000 kcal"
    },
    "Thai Chicken Biryani": {
        "100g": "400-450 kcal",
        "250g": "1000-1125 kcal",
        "500g": "2000-2250 kcal",
        "1kg": "4000-4500 kcal"
    },
    "Thai Pork Leg Stew": {
        "100g": "350-400 kcal",
        "250g": "875-1000 kcal",
        "500g": "1750-2000 kcal",
        "1kg": "3500-4000 kcal"
    }
}


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
                predicted_label = labels[result_index]
                
                # Get calorie info for the predicted label
                calories = calorie_info.get(predicted_label, "N/A")
                
                st.success(f"The model predicts this is: **{predicted_label}**")
                st.info(f"Estimated calories: **{calories} kcal**")
                
                # Confidence Bar Chart
                st.write("Confidence Levels:")
                
                # Create a DataFrame for plotting
                confidence_df = pd.DataFrame({
                    'Label': labels,
                    'Confidence': predictions[0] * 100
                })
                
                st.bar_chart(confidence_df.set_index('Label'))
