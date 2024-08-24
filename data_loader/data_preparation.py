# data_preparation.py
import tensorflow as tf
from data_loader import load_data, create_generators, custom_generator, dataset_from_generator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data():
    df = load_data()
    train_df = df.sample(frac=1, random_state=0).sample(frac=0.8, random_state=0)
    test_df = df.drop(train_df.index)

    # Print and log the sizes of the training and test datasets
    print(f"Training size: {len(train_df)} images")
    print(f"Test size: {len(test_df)} images")
    logger.info(f"Training size: {len(train_df)} images")
    logger.info(f"Test size: {len(test_df)} images")

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

    return train_df, test_df, train_dataset, test_dataset