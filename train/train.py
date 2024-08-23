import sys
import os

# Add the directory containing data_loader.py to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from data_loader import load_data, create_generators, custom_generator, dataset_from_generator
from model import AgeGenderModel
from utils.plotting import plot_statistics
from tensorflow.keras.callbacks import TensorBoard
import datetime

def main():
    df = load_data()
    train_df = df.sample(frac=1, random_state=0).sample(frac=0.8, random_state=0)
    test_df = df.drop(train_df.index)

    train_generator, test_generator = create_generators(train_df, test_df)

    output_signature = (
        tf.TensorSpec(shape=(None, 200, 200, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    )

    train_dataset = dataset_from_generator(lambda: custom_generator(train_generator), output_signature)
    test_dataset = dataset_from_generator(lambda: custom_generator(test_generator), output_signature)

    model = AgeGenderModel()
    model.model.summary()

    model.compile_model(
        optimizer='adamW',
        loss={'age': 'mse', 'gender': 'binary_crossentropy'},
        metrics={'age': 'mse', 'gender': 'accuracy'}
    )

    batch_size = 32
    train_steps = len(train_df) // batch_size
    test_steps = len(test_df) // batch_size

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.train_model(
        train_dataset,
        validation_data=test_dataset,
        epochs=10,
        batch_size=batch_size,
        callbacks=[tensorboard_callback]
    )

    plot_statistics(history)

    model.save_model('age_gender_model.h5')

if __name__ == "__main__":
    main()