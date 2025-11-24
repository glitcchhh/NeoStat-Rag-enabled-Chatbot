# models/embeddings.py
# Embedding model wrapper. Keep model definitions here and call from app.py.
from sentence_transformers import SentenceTransformer
import os
from config.config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {model_name}: {e}")


        def embed(self, texts):
            """Return embeddings for a list of texts."""
            try:
                if isinstance(texts, str):
                    texts = [texts]
                emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return emb
            except Exception as e:
                raise


# Optionally add provider-specific embedding classes (OpenAI) if you want to use cloud embeddings.