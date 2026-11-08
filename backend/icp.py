"""Part 4a: ICP discovery (requirements A2, A3, A5).

Weighted KERNEL k-means on the clients. The kernel is the same weighted-block cosine similarity used
later to assign businesses to profiles and to rank lookalikes, so formation and assignment use ONE
metric (A3). Turnover enters as sample weights (A2). The number of profiles (1..6) is chosen by
bootstrap stability + cohesion gain; "one ICP" is returned when the client list is homogeneous.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def adjusted_rand_score(a, b) -> float:
    """Same integer formula as sklearn.metrics.adjusted_rand_score (same result, tested), without its input validation: about 40x faster."""
    _, a = np.unique(a, return_inverse=True); _, b = np.unique(b, return_inverse=True)
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    c = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(np.int64)
    n = int(c.sum()); nr, nk = c.sum(axis=1), c.sum(axis=0)
    sq = int((c * c).sum())
    fp = int((c @ nk).sum()) - sq; fn = int((c.T @ nr).sum()) - sq
    tp = sq - n; tn = n * n - fp - fn - sq
    if fn == 0 and fp == 0:
        return 1.0
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def kernel_kmeans(K, w, k, rng, n_init=6, max_iter=60, active=None):
    m = len(w)
    active = np.ones(m, bool) if active is None else active
    wf = np.asarray(w, float) * active
    diag = np.diag(K).copy()
    best = None
    for _ in range(n_init):
        c0 = rng.choice(m, p=wf / wf.sum())
        centers = [c0]
        d2 = diag + diag[c0] - 2 * K[:, c0]
        for _ in range(1, k):
            p = wf * np.maximum(d2, 0) + 1e-12
            ci = rng.choice(m, p=p / p.sum())
            centers.append(ci)
            d2 = np.minimum(d2, diag + diag[ci] - 2 * K[:, ci])
        dist = np.stack([diag + diag[c] - 2 * K[:, c] for c in centers], axis=1)
        labels = dist.argmin(1)
        for _ in range(max_iter):
            dist = np.empty((m, k))
            for c in range(k):
                mask = (labels == c) & active
                Wc = wf[mask].sum()
                if Wc <= 0:                      # empty cluster: re-seed on a random active point
                    far = int(rng.choice(np.flatnonzero(active)))
                    dist[:, c] = diag + diag[far] - 2 * K[:, far]
                    continue
                v = wf[mask]
                t2 = K[:, mask] @ v / Wc
                t3 = v @ K[np.ix_(mask, mask)] @ v / Wc ** 2
                dist[:, c] = diag - 2 * t2 + t3
            new = dist.argmin(1)
            if np.array_equal(new, labels):
                break
            labels = new
        cost = float((wf * dist[np.arange(m), labels])[active].sum())
        if best is None or cost < best[0]:
            best = (cost, labels.copy())
    return best[1]


def cohesion(K, w, labels) -> float:
    """Weighted mean leave-one-out average similarity of each point to the rest of its own cluster."""
    w = np.asarray(w, float)
    tot, acc = 0.0, 0.0
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) < 2:
            continue
        Kc = K[np.ix_(idx, idx)]
        wc = w[idx]
        num = Kc @ wc - np.diag(Kc) * wc
        den = wc.sum() - wc
        acc += float((wc * num / np.maximum(den, 1e-12)).sum())
        tot += float(wc.sum())
    return acc / tot if tot else 0.0


def bootstrap_stability(K, w, groups, k, full_labels, B, rng):
    """Resample VATs with replacement, re-cluster, compare with the full-data partition.
    Returns (mean ARI, per-cluster mean Jaccard)."""
    ug, inv = np.unique(groups, return_inverse=True)
    aris, jac = [], {c: [] for c in np.unique(full_labels)}
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        mult = cnt[inv]
        lab = kernel_kmeans(K, w * mult, k, rng, n_init=3, active=mult > 0)
        aris.append(adjusted_rand_score(full_labels, lab))
        for c in jac:
            A = full_labels == c
            jac[c].append(max(((A & (lab == j)).sum() / max((A | (lab == j)).sum(), 1)) for j in np.unique(lab)))
    return float(np.mean(aris)), {int(c): float(np.mean(v)) for c, v in jac.items()}


@dataclass
class ICPResult:
    k: int
    labels: np.ndarray
    stability: dict                 # cluster -> mean Jaccard
    k_table: list = field(default_factory=list)
    rule: str = ""


REF_N_INIT = 12          # restarts for the reference clustering of each k (the bootstrap replicates keep 3)


def discover(sub_model, w, groups, k_max=6, B=30, seed=42, min_gain=0.02, min_ari=0.6,
             min_size_frac=0.05, force_k=None) -> ICPResult:
    """Pick the number of profiles. k is eligible when bootstrap ARI >= min_ari, every profile has
    >= max(8, 5% of clients) shops and cohesion gain over k=1 >= min_gain. The eligible k with the highest
    bootstrap ARI wins (smallest k within 0.03 of the best); none eligible -> ONE ICP."""
    m = sub_model.n
    K = sub_model.sim_to(np.arange(m)).astype(np.float64)
    w = np.asarray(w, float)
    base = cohesion(K, w, np.zeros(m, int))
    table = [{"k": 1, "ari": 1.0, "gain": 0.0, "min_size": m, "eligible": True}]
    cands = {}
    min_size = max(8, int(np.ceil(min_size_frac * m)))
    ks = [force_k] if force_k else range(2, k_max + 1)
    for k in ks:
        if k == 1:
            continue
        # Every k gets its own random stream and a thorough reference clustering. Before, one stream was shared, so the partition
        # tried for k=3 depended on how many bootstraps k=2 had used: 30 bootstraps found a poor k=3 and wrongly answered "2 profiles".
        rng = np.random.default_rng([seed, k])
        labels = kernel_kmeans(K, w, k, rng, n_init=REF_N_INIT)
        sizes = np.bincount(labels, minlength=k)
        gain = cohesion(K, w, labels) - base
        ari, jac = bootstrap_stability(K, w, groups, k, labels, B, rng) if B else (1.0, {c: 1.0 for c in range(k)})
        ok = bool(force_k) or (ari >= min_ari and sizes.min() >= min_size and gain >= min_gain)
        table.append({"k": k, "ari": round(ari, 3), "gain": round(float(gain), 4),
                      "min_size": int(sizes.min()), "eligible": ok})
        if ok:
            cands[k] = (ari, labels, jac)
    if not cands:
        return ICPResult(1, np.zeros(m, int), {0: 1.0}, table,
                         "Κανένα k≥2 δεν πληροί τα όρια σταθερότητας, μεγέθους και συνοχής → ένα ICP")
    top = max(v[0] for v in cands.values())
    kbest = min(k for k, v in cands.items() if v[0] >= top - 0.03)      # most stable; fewer profiles on ties
    _, labels, jac = cands[kbest]
    order = np.argsort(-np.bincount(labels, weights=w, minlength=kbest))   # profile 1 = biggest turnover
    remap = {int(o): i for i, o in enumerate(order)}
    labels = np.array([remap[int(x)] for x in labels])
    jac = {remap[c]: v for c, v in jac.items()}
    return ICPResult(kbest, labels, jac, table, f"Επιλέχθηκε k={kbest}: η μεγαλύτερη σταθερότητα (bootstrap ARI) ανάμεσα στα επιλέξιμα k (ισοπαλία ±0,03 → λιγότερα προφίλ)")
