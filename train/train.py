import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from logger import setup_logger
from tensorflow.keras.callbacks import TensorBoard
from utils.plotting import plot_statistics
from model import AgeGenderModel
from utils.util import save_model
from utils.config_loader import load_config
from utils.tf_config_loader import configure_tensorflow
from data_loader.data_preparation import prepare_data
from utils.device_manager import get_device
import tensorflow as tf

def main():
    # Load configuration
    config = load_config()

    # Configure TensorFlow
    configure_tensorflow(config["cpu_threads"], config["gpu_threads"])

    logger, log_dir = setup_logger()

    logger.info("Loading and preparing data...")
    train_df, test_df, train_dataset, test_dataset = prepare_data()

    logger.info("Initializing model...")
    model = AgeGenderModel()
    model.model.summary(print_fn=logger.info)

    logger.info("Compiling model...")
    model.compile_model(
        optimizer=config["optimizer"],
        loss=config["loss"],
        metrics=config["metrics"]
    )

    train_steps = len(train_df) // config["batch_size"]
    test_steps = len(test_df) // config["batch_size"]
    print('Train steps: {}, Test steps: {}'.format(train_steps, test_steps))

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    device = get_device()

    with tf.device(device):
        logger.info("Starting model training...")
        history = model.train_model(
            train_dataset,
            validation_data=test_dataset,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=[tensorboard_callback]
        )

    logger.info("Plotting training statistics...")
    plot_statistics(history)

    save_model(model, logger)

    logger.info("Training process completed.")


if __name__ == "__main__":
    main()