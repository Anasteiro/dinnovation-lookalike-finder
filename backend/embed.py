"""KAD description embeddings. We only embed the UNIQUE KAD descriptions (hundreds, not 24k rows),
so comparing several embedding models is cheap (brief section 5.3).

backends:
  tfidf               char n-gram TF-IDF + SVD, fully offline, used for tests / fallback
  st:<hf-model-name>  sentence-transformers model run locally (e.g. st:intfloat/multilingual-e5-base)
"""
from __future__ import annotations
import hashlib
import numpy as np
from .config import CACHE_DIR


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _tfidf(texts, dim=64):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit_transform(texts)
    k = max(2, min(dim, X.shape[0] - 1, X.shape[1] - 1))
    return TruncatedSVD(n_components=k, random_state=0).fit_transform(X)


def _st(texts, name, batch_size=64):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(name, device=device)
    prefix = "query: " if "e5" in name.lower() else ""
    emb = model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def embed_texts(texts: list[str], backend: str = "tfidf", use_cache: bool = True) -> np.ndarray:
    key = hashlib.md5((backend + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = CACHE_DIR / f"emb_{key}.npy"
    if use_cache and path.exists():
        return np.load(path)
    if backend == "tfidf":
        E = _tfidf(texts)
    elif backend.startswith("st:"):
        E = _st(texts, backend[3:])
    else:
        raise ValueError(f"unknown embedding backend {backend}")
    E = _l2(np.asarray(E, dtype=np.float32))
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, E)
    return E
