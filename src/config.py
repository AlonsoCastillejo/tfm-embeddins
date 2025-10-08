import os
from dotenv import load_dotenv

# Cargamos variables de entorno
load_dotenv()

class Config:
    # API Key de OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    EMBEDDING_MODEL_SMALL = "text-embedding-3-small"
    EMBEDDING_MODEL_LARGE = "text-embedding-3-large"
    
    # ChromaDB
    CHROMADB_PATH = "chromadb"
    COLLECTION_NAME = "pisos"
    
    # App
    MAX_RESULTS = 10
    DEFAULT_RESULTS = 3

    