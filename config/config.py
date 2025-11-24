# config/config.py
# Keep secrets out of version control. Read from environment variables.
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # e.g. for text-gen / embeddings
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



# App settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = "perplexity"


# Vector store path
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store.pkl")