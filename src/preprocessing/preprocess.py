import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(data):
    if data.duplicated().sum()>0:
        data=data.drop_duplicates()
    datagen=ImageDataGenerator(rescale=1./255)
    train_datagen_flow=datagen.flow_from_dataframe(
    dataframe=data,
    directory='./files/datasets/input/faces/final_files/',
    target_size=(224,224),
    batch_size=16,
    class_mode='raw',
    x_col='file_name',
    y_col='real_age',
    subset='training',
    seed=12345)
    features,target=next(train_datagen_flow)
    return features,target