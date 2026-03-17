import os
import logging
import pickle
from typing import List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, INDEX_DIR, TOP_K
from pdf_extractor import Chunk

logger = logging.getLogger(__name__)


class FAISSIndex:
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        self.model = SentenceTransformer(EMBED_MODEL)

        logger.info(f"Embedding {len(chunks)} chunks...")
        embeddings = self.model.encode(
            [c.text for c in chunks],
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks

        logger.info(f"Index built: {self.index.ntotal} vectors, dim={dim}")

    def save(self, index_dir: str = INDEX_DIR) -> None:
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Index saved to {index_dir}")

    def load(self, index_dir: str = INDEX_DIR) -> bool:
        idx_path = os.path.join(index_dir, "faiss.index")
        chk_path = os.path.join(index_dir, "chunks.pkl")
        if not (os.path.exists(idx_path) and os.path.exists(chk_path)):
            return False
        self.index = faiss.read_index(idx_path)
        with open(chk_path, "rb") as f:
            self.chunks = pickle.load(f)
        if self.model is None:
            logger.info(f"Loading embedding model: {EMBED_MODEL}")
            self.model = SentenceTransformer(EMBED_MODEL)
        logger.info(f"Index loaded: {self.index.ntotal} vectors")
        return True

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, ids = self.index.search(q_emb, top_k)
        return [
            (self.chunks[idx], float(score))
            for score, idx in zip(scores[0], ids[0])
            if idx >= 0
        ]

    def search_with_company_filter(
        self, query: str, company_name: str, top_k: int = TOP_K
    ) -> List[Tuple[Chunk, float]]:
        candidates = self.search(query, top_k=top_k * 4)
        name_lower = company_name.lower()
        hits_with = [(c, s) for c, s in candidates if name_lower in c.text.lower()]
        hits_without = [(c, s) for c, s in candidates if name_lower not in c.text.lower()]
        return (hits_with + hits_without)[:top_k]
