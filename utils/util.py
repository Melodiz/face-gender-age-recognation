import tensorflow as tf
import torch
import os

def check_gpu():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def save_model(model, logger, model_path='models/age_gender_model.h5'):
    # Ensure the models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        logger.info(f"Saving model to '{model_path}'...")
        model.save_model(model_path)
    except Exception as e:
        logger.error(f"Failed to save model as '{model_path}': {e}")
        logger.info("Saving model to 'age_gender_model.tf' and 'age_gender_model.h5'...")
        model.save_model('age_gender_model.tf')
        model.save_model('age_gender_model.h5')

