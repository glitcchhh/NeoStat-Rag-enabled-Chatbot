# utils/embeddings_utils.p
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.embeddings import EmbeddingModel
from config.config import VECTOR_STORE_PATH


EMB = EmbeddingModel()




def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """Simple text chunker by characters (works reasonably for many doc types).
    Returns list of chunks."""
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks




def build_vector_store(docs: list, meta: list = None, path: str = VECTOR_STORE_PATH):
   
    try:
        if meta is None:
            meta = [None] * len(docs)
        embeddings = EMB.embed(docs)
        store = {"embeddings": embeddings, "docs": docs, "meta": meta}
        with open(path, "wb") as f:
            pickle.dump(store, f)
        return path
    except Exception as e:
        raise




def load_vector_store(path: str = VECTOR_STORE_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        store = pickle.load(f)
    return store




def retrieve(query: str, k: int = 4, path: str = VECTOR_STORE_PATH):
    """Return top-k docs (text, score, meta) relevant to query using cosine similarity."""
    try:
        store = load_vector_store(path)
        if not store:
            return []
        q_emb = EMB.embed(query)[0]
        embeddings = store["embeddings"]
        # Ensure shapes
        if len(embeddings.shape) == 1:
            embeddings = np.expand_dims(embeddings, 0)
        sims = cosine_similarity([q_emb], embeddings)[0]
        idxs = np.argsort(sims)[::-1][:k]
        results = []
        for i in idxs:
            results.append({"doc": store["docs"][i], "score": float(sims[i]), "meta": store["meta"][i]})
        return results
    except Exception as e:
        print("Retrieval error:", e)
        return []