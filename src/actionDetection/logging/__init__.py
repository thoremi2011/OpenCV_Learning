import os
import sys
import logging
from pathlib import Path

# Configure logging directory
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Logging format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),  # Write to file
        logging.StreamHandler(sys.stdout)   # Show in console
    ]
)

# Create a logger that can be used throughout the project
logger = logging.getLogger("actionDetectionLogger")
