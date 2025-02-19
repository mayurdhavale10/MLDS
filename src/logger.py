import logging
import os
from datetime import datetime

# Define the log file name based on the current datetime.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the logs directory if it doesn't exist.
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Define the complete path for the log file.
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Set up basic logging configuration.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
