# tf_config.py
import os
import tensorflow as tf

def configure_tensorflow(cpu_threads, gpu_threads):
    # Set environment variable to limit CPU usage
    if cpu_threads == -1:
        cpu_threads = os.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)

    # Configure TensorFlow to use a limited number of CPU threads
    tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)

    # Configure TensorFlow to use all available GPUs if gpu_threads is -1
    if gpu_threads == -1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    else:
        tf.config.threading.set_intra_op_parallelism_threads(gpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(gpu_threads)