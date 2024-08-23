import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def load_data(folder_path='data/UTKFace'):
    age = []
    gender = []
    img_path = []

    for file in os.listdir(folder_path):
        age.append(int(file.split('_')[0]))
        gender.append(int(file.split('_')[1]))
        img_path.append(file)

    df = pd.DataFrame({'age': age, 'gender': gender, 'img': img_path})
    return df

def create_generators(train_df, test_df, folder_path='data/UTKFace'):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=30,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(train_df,
                                                        directory=folder_path,
                                                        x_col='img',
                                                        y_col=['age', 'gender'],
                                                        target_size=(200, 200),
                                                        class_mode='raw')

    test_generator = test_datagen.flow_from_dataframe(test_df,
                                                      directory=folder_path,
                                                      x_col='img',
                                                      y_col=['age', 'gender'],
                                                      target_size=(200, 200),
                                                      class_mode='raw')
    return train_generator, test_generator

def custom_generator(generator):
    for x, y in generator:
        age = np.array([label[0] for label in y])
        gender = np.array([label[1] for label in y])
        gender = gender.reshape((-1, 1))
        yield x, (age, gender)

def dataset_from_generator(generator, output_signature):
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)