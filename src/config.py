import os

from dotenv import load_dotenv


load_dotenv()


DISTANCE_THRESHOLD = 1.2

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)