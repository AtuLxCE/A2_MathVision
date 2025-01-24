# utils/helpers.py

from huggingface_hub import login
from config.settings import HF_TOKEN

def login_to_huggingface():
    """Log in to Hugging Face."""
    login(HF_TOKEN)