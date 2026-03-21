from openai import OpenAI
from app.config import OPENAI_API_KEY

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY manquante. Vérifie ton fichier .env")

client = OpenAI(api_key=OPENAI_API_KEY)