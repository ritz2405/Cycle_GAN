from keras.models import load_model
import tensorflow as tf
import keras_contrib
import keras
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from sklearn.utils import resample
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.utils import img_to_array
from keras.utils import load_img
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import numpy as np
import streamlit as st
from PIL import Image


def get_model():
    # Load your pre-trained model
    custom_objects = {'InstanceNormalization': keras_contrib.layers.normalization.instancenormalization.InstanceNormalization}
    photo_to_monet = load_model('g_model_BtoA_00005360.h5', custom_objects) # Replace with actual model loading code
    monet_to_photo = load_model('g_model_AtoB_00005360.h5', custom_objects)
    opt = Adam(lr=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
    photo_to_monet.compile(loss=['mse', 'mae', 'mae', 'mae'], 
               loss_weights=[1, 5, 10, 10], optimizer=opt)
    monet_to_photo.compile(loss=['mse', 'mae', 'mae', 'mae'], 
               loss_weights=[1, 5, 10, 10], optimizer=opt)
    return photo_to_monet

def apply_monet_style(model, img_array):
    # Preprocess the input image array as required by your model
    # Example: resize, scale pixel values, etc.
    # img_array = preprocess_function(img_array)  # Uncomment and define preprocess_function if needed
    
    # Apply the Monet style transformation using the model
    transformed_img_array = model.predict(img_array)
    
    # Post-process the output array if necessary
    # Example: remove batch dimension, scale pixel values back, etc.
    transformed_img_array = np.squeeze(transformed_img_array)
    
    return transformed_img_array
