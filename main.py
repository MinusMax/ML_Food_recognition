import streamlit as st
import tensorflow as tf
import numpy as np

# Load the model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.h5")

# TensorFlow Model Prediction
def model_prediction(test_image, model):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction", "Calorie Calculation"])

# Load the model
model = load_model()

# Main Page
if app_mode == "Home":
    st.header("Thai Food Recognition System")
    image_path = "Thaifood_banner.jpg"  # Update to your image path
    st.image(image_path)

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following Thai food items:")
    st.code("""Baked Prawns With Vermicelli,
    Banana in Coconut Milk,
    Chicken Green Curry,
    Chicken Mussaman Curry,
    Coconut Rice Pancake,
    Curried Noodle Soup with Chicken,
    Egg and Pork in Sweet Brown Sauce,
    Egg with Tamarind Sauce,
    Fried Cabbage with Fish Sauce,
    Fried Rice,
    Grilled River Prawn,
    Mango Sticky Rice,
    Omelet,
    Pork Chopped Tofu Soup,
    Pork Curry With Morning Glory,
    Shrimp Fried Rice,
    Shrimp Paste Fried Rice,
    Sour Soup,
    Spicy Mixed Vegetable Soup,
    Steamed Capon in Flavored Rice,
    Stir Fried Chicken with Chestnuts,
    Stir Fried Rice Noodles with Chicken,
    Stuffed Bitter Gourd Broth,
    Thai Chicken Biryani,
    Thai Pork Leg Stew.
    """)
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=400, use_column_width=True)
        # Predict button
        if st.button("Predict"):
            st.snow()
            result_index = model_prediction(test_image, model)
            # Reading Labels
            with open("labels.txt") as f:
                content = f.readlines()
            label = [i.strip() for i in content]
            st.success("Model is predicting it's a {}".format(label[result_index]))
            st.session_state.prediction_result = label[result_index]
            st.session_state.test_image = test_image

# Calorie Calculation Page
elif app_mode == "Calorie Calculation":
    st.header("Calorie Calculation")
    if 'prediction_result' in st.session_state and 'test_image' in st.session_state:
        st.image(st.session_state.test_image, width=400, use_column_width=True)
        st.write("Predicted Food Item: {}".format(st.session_state.prediction_result))
        
        weight = st.number_input("Enter weight in grams:", min_value=1, max_value=5000, value=100)
        calorie_per_100g = 10  # Set according to your specific food item
        calories = (weight / 100) * calorie_per_100g

        st.write(f"Calories for {weight}g: {calories:.2f} kcal")
        st.success("Calculated Calories: {} kcal".format(calories))
    else:
        st.warning("Please go to the Prediction page first and make a prediction.")

