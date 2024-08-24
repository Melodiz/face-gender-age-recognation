# device_manager.py
import tensorflow as tf

def get_device():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        device = '/GPU:0'
    else:
        print(f'Warning: No GPU found. Running on CPU instead...')
        device = '/CPU:0'
    return device