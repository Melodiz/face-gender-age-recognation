import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def load_data(folder_path='data/', extra_size=True):
    if not extra_size: return load_UTKFace_data()
    else: 
        UTK_df = load_UTKFace_data()
        AgeDB_df = load_AgeDB_data()
        return pd.concat([UTK_df, AgeDB_df], ignore_index=True)

def load_UTKFace_data(folder_path='data/UTKFace'):
    age = []
    gender = []
    img_path = []

    for file in os.listdir(folder_path):
        age.append(int(file.split('_')[0]))
        gender.append(int(file.split('_')[1]))
        img_path.append('UTKFace/'+file)

    df = pd.DataFrame({'age': age, 'gender': gender, 'img': img_path})
    return df

def load_AgeDB_data(folder_path='data/AgeDB'):
    lable_to_label = {'f': 1, 'm': 0}
    age = []
    gender = []
    img_path = []

    for file in os.listdir(folder_path):
        age.append(int(file.split('_')[-2]))
        gender.append(lable_to_label[file.split('_')[-1].split('.')[0]])
        img_path.append('AgeDB/'+file)
    df = pd.DataFrame({'age': age, 'gender': gender, 'img': img_path})
    return df

def create_generators(train_df, test_df, folder_path='data/'):
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

    # Print invalid filenames
    invalid_train_filenames = set(train_df['img']) - set(train_generator.filenames)
    invalid_test_filenames = set(test_df['img']) - set(test_generator.filenames)
    if invalid_train_filenames:
        print(f"Invalid train filenames: {invalid_train_filenames}")
    if invalid_test_filenames:
        print(f"Invalid test filenames: {invalid_test_filenames}")

    return train_generator, test_generator
def custom_generator(generator):
    for x, y in generator:
        age = np.array([label[0] for label in y])
        gender = np.array([label[1] for label in y])
        gender = gender.reshape((-1, 1))
        yield x, (age, gender)


def dataset_from_generator(generator, output_signature):
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)