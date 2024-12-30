# Configuration settings (e.g., API keys, DB URI)
import os
from dotenv import find_dotenv, load_dotenv
# Environment setup
dotenv_path = find_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")