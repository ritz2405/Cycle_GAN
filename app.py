import streamlit as st
from PIL import Image
import numpy as np
import monet_model  # Ensure this is the correct import path for your module
from keras.utils import img_to_array

# Load the pre-trained model
model = monet_model.get_model()

def transform_image(image):
    # Convert the image to an array
    image = image.resize((256, 256))
    image = img_to_array(image)
    img_array = np.array([image])
    img_array = (img_array - 127.5)/127.5
    
    # Apply the Monet style transformation
    transformed_img = monet_model.apply_monet_style(model, img_array)
    
    # Convert the result back to an image
    transformed_img = Image.fromarray((transformed_img * 255).astype(np.uint8))  # Adjust scaling if needed
    return transformed_img

st.title("Turn Your Photo into a Monet Painting")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("")
    st.write("Transforming...")
    
    # Transform the image
    transformed_image = transform_image(image)
    
    # Display the transformed image
    st.image(transformed_image, caption='Monet Style Image', use_column_width=True)
