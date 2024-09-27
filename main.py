import streamlit as st
import tensorflow as tf
import numpy as np

# Load the model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model2.h5")

# TensorFlow Model Prediction
def model_prediction(test_image, model):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Calorie information dictionary
calorie_info = {
    "Baked Prawns With Vermicelli": 150,
    "Banana in Coconut Milk": 120,
    "Chicken Green Curry": 150,
    "Chicken Mussaman Curry": 100,
    "Coconut Rice Pancake": 200,
    "Curried Noodle Soup with Chicken": 130,
    "Egg and Pork in Sweet Brown Sauce": 180,
    "Egg with Tamarind Sauce": 160,
    "Fried Cabbage with Fish Sauce": 80,
    "Fried Egg": 155,
    "Fried Rice": 150,
    "Grilled River Prawn": 120,
    "Mango Sticky Rice": 160,
    "Omelet": 150,
    "Pork Chopped Tofu Soup": 120,
    "Pork Curry With Morning Glory": 140,
    "Shrimp Fried Rice": 170,
    "Shrimp Paste Fried Rice": 180,
    "Sour Soup": 90,
    "Spicy Mixed Vegetable Soup": 60,
    "Steamed Capon in Flavored Rice": 200,
    "Stir Fried Chicken with Chestnuts": 160,
    "Stir Fried Rice Noodles with Chicken": 170,
    "Stuffed Bitter Gourd Broth": 80,
    "Thai Chicken Biryani": 200,
    "Thai Pork Leg Stew": 250,
    "Stir-Fried Chinese Morning Glory": 40,
    "Fried noodle Thai style with prawns": 190,
    "Stir fried Thai basil with minced pork": 220,
    "Fried Noodle in Soy Sauce": 180,
    "Stir-fried Pumpkin with Eggs": 100,
    "Stir-Fried Eggplant with Soybean Paste Sauce": 140,
    "Stir Fried Clams with Roasted Chili Paste": 120,
    "Golden Egg Yolk Threads": 130,
    "Chicken Penang": 160,
    "Thai Wing Beans Salad": 80,
    "Spicy Glass Noodle Salad": 140,
    "Spicy minced pork salad": 160,
    "Egg custard in pumpkin": 200,
    "Tapioca Balls with Pork Filling": 210,
    "Green Papaya Salad": 60,
    "Thai-Style Grilled Pork Skewers": 170,
    "Pork Satay with Peanut Sauce": 180,
    "Steamed Fish with Curry Paste": 130
}


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
    st.write("Welcome to the Thai Food Recognition System! Use the sidebar to navigate through the app.")

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following Thai food items:")
    st.write("- Baked Prawns With Vermicelli")
    st.write("- Banana in Coconut Milk")
    st.write("- Chicken Green Curry")
    st.write("- Chicken Mussaman Curry")
    st.write("- Coconut Rice Pancake")
    st.write("- Curried Noodle Soup with Chicken")
    st.write("- Egg and Pork in Sweet Brown Sauce")
    st.write("- Egg with Tamarind Sauce")
    st.write("- Fried Cabbage with Fish Sauce")
    st.write("- Fried Egg")
    st.write("- Fried Rice")
    st.write("- Grilled River Prawn")
    st.write("- Mango Sticky Rice")
    st.write("- Omelet")
    st.write("- Pork Chopped Tofu Soup")
    st.write("- Pork Curry With Morning Glory")
    st.write("- Shrimp Fried Rice")
    st.write("- Shrimp Paste Fried Rice")
    st.write("- Sour Soup")
    st.write("- Spicy Mixed Vegetable Soup")
    st.write("- Steamed Capon in Flavored Rice")
    st.write("- Stir Fried Chicken with Chestnuts")
    st.write("- Stir Fried Rice Noodles with Chicken")
    st.write("- Stuffed Bitter Gourd Broth")
    st.write("- Thai Chicken Biryani")
    st.write("- Thai Pork Leg Stew")
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

        # Get the calorie information for the predicted food item
        food_item = st.session_state.prediction_result
        calorie_per_100g = calorie_info.get(food_item, 0)

        weight = st.number_input("Enter weight in grams:", min_value=1, max_value=5000, value=100)
        calories = (weight / 100) * calorie_per_100g  # Calculate calories

        st.write(f"Calories for {weight}g: {calories:.2f}")
        st.success("Calculated Calories: {} calories".format(calories))
    else:
        st.warning("Please go to the Prediction page first and make a prediction.")
