"""Text -> vector. One class; swap the model in config.py."""
import numpy as np
import config


class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer   # imported lazily: slow import
        self.model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)

    def _encode(self, texts, prefix):
        if config.E5_PREFIXES:
            texts = [prefix + t for t in texts]
        vecs = self.model.encode(texts, batch_size=config.EMBED_BATCH,
                                 normalize_embeddings=True, show_progress_bar=len(texts) > 50)
        return np.asarray(vecs, dtype=np.float32)

    def passages(self, texts):
        return self._encode(texts, "passage: ")

    def query(self, text):
        return self._encode([text], "query: ")[0]
