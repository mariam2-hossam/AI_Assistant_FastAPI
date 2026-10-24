import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
URL = "https://openrouter.ai/api/v1/chat/completions"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
