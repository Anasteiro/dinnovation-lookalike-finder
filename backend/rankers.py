"""Scorers plugged into the evaluation harness (and later into the product)."""
from __future__ import annotations
import numpy as np


def revenue_weights(rev, mode="log", clip_q=(0.05, 0.95)) -> np.ndarray:
    """Documented use of turnover (A2): log-transform, winsorise, normalise to mean 1."""
    rev = np.asarray(rev, float)
    if mode == "none" or np.all(np.isnan(rev)):
        return np.ones(len(rev))
    x = np.log1p(np.clip(np.nan_to_num(rev, nan=np.nanmedian(rev)), 0, None)) if mode == "log" \
        else np.sqrt(np.clip(np.nan_to_num(rev, nan=np.nanmedian(rev)), 0, None))
    lo, hi = np.quantile(x, clip_q)
    x = np.clip(x, lo, hi)
    return x / x.mean()


def centroid_scorer(model, weights=None):
    return lambda train, w: model.centroid_scores(train, w, weights)


def knn_scorer(model, m=5, alpha=0.0, weights=None):
    return lambda train, w: model.knn_scores(train, w, m=m, alpha=alpha, weights=weights)


def combine_profiles(raw: np.ndarray, loo_sorted: list, how="raw") -> np.ndarray:
    """Merge per-profile scores (n x k) into one global score per business.
    raw  : max of raw scores (favours the tightest profile)
    z    : each profile's scores standardised over the whole market, then max
    ecdf : position among the profile's own members (leave-one-out), raw score only breaks ties"""
    if how == "raw":
        return raw.max(axis=1)
    if how == "z":
        mu, sd = raw.mean(axis=0), raw.std(axis=0) + 1e-9
        return ((raw - mu) / sd).max(axis=1)
    if how == "ecdf":
        e = np.stack([np.searchsorted(l, raw[:, c], side="right") / max(len(l), 1) for c, l in enumerate(loo_sorted)], axis=1)
        return (e + 1e-3 * raw).max(axis=1)
    raise ValueError(how)


def profile_scorer(model, k=3, blend=0.0, m_knn=5, seed=42, how="raw", weights=None):
    """The product's method inside the CV harness: cluster the TRAINING clients into k profiles with
    weighted kernel k-means, score every business against each profile, merge with `how`."""
    from . import icp, lookalike as L

    def fn(train, w):
        w = np.asarray(w, float)
        sub = model.subset(train)
        K = sub.sim_to(np.arange(sub.n), weights).astype(np.float64) if weights else sub.sim_to(np.arange(sub.n)).astype(np.float64)
        kk = min(k, max(1, sub.n // 8))
        labels = icp.kernel_kmeans(K, w, kk, np.random.default_rng(seed)) if kk > 1 else np.zeros(sub.n, int)
        cols, loos = [], []
        for c in range(kk):
            mem = np.flatnonzero(labels == c)
            if len(mem) < 2:
                continue
            sc, _ = L.profile_scores(model, train[mem], w[mem], m_knn, blend, weights)
            cols.append(sc)
            loos.append(np.sort(L.member_loo_scores(sub.subset(mem), w[mem], m_knn, blend, weights)))
        return combine_profiles(np.stack(cols, axis=1), loos, how)
    return fn


# ---------------------------------------------------------------- PU-learning (idea from the second developer's engine)
def dense_features(model, weights=None, dims=96, seed=0):
    """Weighted blocks -> one sparse matrix -> truncated SVD -> standardised dense features (N x dims) for the PU classifier."""
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    w, tot = model._w(weights)
    X = sp.hstack([sp.csr_matrix(model.blocks[b]) * np.sqrt(wb / tot) for b, wb in w.items()], format="csr")
    d = min(dims, X.shape[1] - 1)
    Z = TruncatedSVD(d, random_state=seed).fit_transform(X)
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-6)


def pu_scores(Z, pos, w, l2=5e-3, boost=8.0, steps=150, neg=None, neg_w=0.0):
    """Positive-Unlabeled logistic regression: clients = positives, rest of the market = unlabeled.
    Plain numpy (Adam), no PyTorch needed; same loss as the second developer's engine.
    neg / neg_w: optional rows a person has confirmed are NOT a fit; they keep label 0 and just count `neg_w` times more
    (feedback re-ranking). With neg=None the result is bit-for-bit the plain PU score."""
    Z = np.asarray(Z, np.float32)                 # float32: about twice as fast on a laptop CPU
    n, d = Z.shape
    y = np.zeros(n, np.float32); y[pos] = 1
    sw = np.ones(n, np.float32); sw[pos] = boost * np.asarray(w, np.float32)
    if neg is not None and len(neg):
        sw[neg] = neg_w
    W = np.zeros(d, np.float32); b = np.float32(-3.0)
    mW, vW, mb, vb = np.zeros(d, np.float32), np.zeros(d, np.float32), 0.0, 0.0
    for t in range(1, steps + 1):
        p = 1 / (1 + np.exp(-(Z @ W + b)))
        g = sw * (p - y) / n
        gW = Z.T @ g + 2 * l2 * W; gb = g.sum()
        mW = .9 * mW + .1 * gW; vW = .999 * vW + .001 * gW ** 2
        mb = .9 * mb + .1 * gb; vb = .999 * vb + .001 * gb ** 2
        W -= .05 * (mW / (1 - .9 ** t)) / (np.sqrt(vW / (1 - .999 ** t)) + 1e-8)
        b -= .05 * (mb / (1 - .9 ** t)) / (np.sqrt(vb / (1 - .999 ** t)) + 1e-8)
    return Z @ W + b


def pu_scorer(model, Z=None, **kw):
    Zm = dense_features(model) if Z is None else Z
    return lambda train, w: pu_scores(Zm, train, np.asarray(w, float) / np.mean(w), **kw)


def _pct(s):
    return np.argsort(np.argsort(s)) / max(len(s) - 1, 1)


def fused_scorer(model, Z=None, w_knn=0.35, w_pu=0.65, m_knn=5, weights=None, **kw):
    """Rank-average of the kNN similarity score and the PU score."""
    Zm = dense_features(model) if Z is None else Z
    def fn(train, w):
        w = np.asarray(w, float)
        knn = model.knn_scores(train, w, m=m_knn, weights=weights)
        pu = pu_scores(Zm, train, w / np.mean(w), **kw)
        return w_knn * _pct(knn) + w_pu * _pct(pu)
    return fn


def merge_proportional(scores: np.ndarray, share: np.ndarray) -> np.ndarray:
    """Proportional-budget merge (idea from the second developer's engine). scores: n x k per-profile scores.
    A business's key = min over profiles of rank_p / share_p, so a profile carrying 2x the (turnover-weighted)
    clients gets 2x the slots in the global top-k. Returns a global score (higher = better)."""
    n, k = scores.shape
    ranks = np.stack([np.argsort(np.argsort(-scores[:, c])) for c in range(k)], axis=1).astype(float)
    return -(ranks / np.maximum(share, 1e-9)[None, :]).min(axis=1)


def profile_pu_scorer(model, Z=None, k=2, w_knn=0.2, w_pu=0.5, w_puall=0.3, m_knn=5, seed=42, weights=None, how="prop", pukw=None):
    """Per-profile PU-learning + kNN, fused per profile, merged with proportional budget."""
    from . import icp, lookalike as L
    Zm = dense_features(model) if Z is None else Z

    def fn(train, w):
        w = np.asarray(w, float)
        sub = model.subset(train)
        K = sub.sim_to(np.arange(sub.n), weights).astype(np.float64)
        kk = min(k, max(1, sub.n // 8))
        labels = icp.kernel_kmeans(K, w, kk, np.random.default_rng(seed)) if kk > 1 else np.zeros(sub.n, int)
        pukw_ = pukw or {}
        puall = _pct(pu_scores(Zm, train, w / w.mean(), **pukw_)) if w_puall else 0.0
        cols, shares = [], []
        for c in range(kk):
            mem = np.flatnonzero(labels == c)
            if len(mem) < 3:
                continue
            knn = _pct(model.knn_scores(train[mem], w[mem], m=m_knn, weights=weights))
            pu = _pct(pu_scores(Zm, train[mem], w[mem] / w[mem].mean(), **pukw_))
            cols.append(w_knn * knn + w_pu * pu + w_puall * puall)
            shares.append(w[mem].sum())
        S = np.stack(cols, axis=1); sh = np.array(shares) / np.sum(shares)
        return merge_proportional(S, sh) if how == "prop" else S.max(axis=1)
    return fn


def rbf_geo(lat, lon, scales=(0.08, 0.5), n_anchor=128, seed=0):
    """Smooth multi-scale location features: Gaussian similarity to anchor points sampled from the market (so shops 100 m
    apart look alike, unlike hard grid cells). Missing GPS -> municipality-free fallback to the mean. Rows are L2-normalised."""
    import pandas as pd
    la = pd.to_numeric(pd.Series(lat), errors="coerce"); lo = pd.to_numeric(pd.Series(lon), errors="coerce")
    la = la.fillna(la.mean()).to_numpy(float); lo = lo.fillna(lo.mean()).to_numpy(float)
    P = np.c_[la, lo * np.cos(np.deg2rad(la.mean()))]
    idx = np.random.default_rng(seed).permutation(len(P))[:min(n_anchor, len(P))]
    d2 = ((P[:, None, :] - P[idx][None, :, :]) ** 2).sum(-1)
    parts = []
    for t in scales:
        g = np.exp(-d2 / (2 * t * t)) + 1e-6
        parts.append(g / np.linalg.norm(g, axis=1, keepdims=True))
    G = np.hstack(parts)
    return G / np.linalg.norm(G, axis=1, keepdims=True)


def pu_features(model, lat, lon):
    """dense_features_v2 with the default settings, computed once per market and kept on the model object.
    It does not depend on the client list, so every analysis after the first reuses it (saves ~4 s per run)."""
    Z = getattr(model, "_pu_features", None)
    if Z is None or Z.shape[0] != len(lat):
        path = _pu_cache_path(model, lat, lon)
        try:
            Z = np.load(path) if path is not None and path.exists() else None
        except Exception:
            Z = None
        if Z is None or Z.shape[0] != len(lat):
            Z = dense_features_v2(model, lat, lon)
            if path is not None:                       # disk cache: the ~4 s SVD is done once per market, not per server start
                try:
                    path.parent.mkdir(parents=True, exist_ok=True); np.save(path, Z)
                except OSError:
                    pass
        model._pu_features = Z
    return Z


def _pu_cache_path(model, lat, lon):
    """Cache file keyed by a fingerprint of the market features (any change in data/settings -> new key)."""
    try:
        import hashlib
        from .config import CACHE_DIR
        h = hashlib.sha1(b"pu_v3|gram|96")
        for name in ("activity", "legal", "age", "struct"):
            m = model.blocks[name]
            h.update(f"{name}:{m.shape}:{getattr(m, 'nnz', m.size)}:{float(abs(m).sum()):.6f}".encode())
        la, lo = np.asarray(lat, float), np.asarray(lon, float)
        h.update(f"{len(la)}:{np.nansum(la):.6f}:{np.nansum(lo):.6f}:{np.isnan(la).sum()}".encode())
        return CACHE_DIR / f"pu_{h.hexdigest()[:16]}.npy"
    except Exception:
        return None


def dense_features_v2(model, lat, lon, blocks=("activity", "legal", "age", "struct"), bw=None, dims=96, seed=0, geo=True):
    """Features for the PU classifier: our activity/legal/age/struct blocks + smooth RBF geography -> SVD -> standardised."""
    import scipy.sparse as sp
    bw = bw or {"activity": .55, "geo": .25, "legal": .12, "age": .08}
    mats = []
    for b in blocks:
        mats.append(sp.csr_matrix(model.blocks[b]) * np.sqrt(bw.get(b, .1)))
    if geo:
        mats.append(sp.csr_matrix(rbf_geo(lat, lon)) * np.sqrt(bw["geo"]))
    X = sp.hstack(mats, format="csr")
    Z = _svd_scores(X, min(dims, X.shape[1] - 1))
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-6)


def _svd_scores(X, k):
    """Top-k (uncentred) principal scores X @ V_k, exact. The matrix is ~95% dense with only a few hundred columns, so the
    Gram matrix X'X (D x D, one BLAS call) + a small eigen-decomposition is ~30x faster than a randomized sparse SVD
    (0.12 s vs 3.6 s on the 24k-business market) and gives the same components (mean |corr| 0.999). Signs are fixed so the
    result is deterministic."""
    Xd = X.toarray().astype(np.float32) if hasattr(X, "toarray") else np.asarray(X, np.float32)
    w, V = np.linalg.eigh((Xd.T @ Xd).astype(np.float64))
    V = V[:, np.argsort(-w)[:k]]
    V *= np.where(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])] < 0, -1.0, 1.0)     # largest |loading| positive
    return Xd @ V.astype(np.float32)
