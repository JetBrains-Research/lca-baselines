import logging
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load env variables from .env
load_dotenv()

# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
