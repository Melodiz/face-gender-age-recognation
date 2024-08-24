import logging
import os
import datetime

def setup_logger(log_dir="logs/fit/"):
    # Create log directory if it doesn't exist
    log_dir = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging configuration
    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    return logger, log_dir