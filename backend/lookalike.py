"""Part 5: scoring of businesses against profiles (A3, A7), 'fits no profile', Greek reasons."""
from __future__ import annotations
import numpy as np
import pandas as pd


def profile_scores(model, members_pos, members_w, m_knn=5, blend=0.0, weights=None):
    """Blend of centroid cosine and mean of the m nearest members. Returns (score, nearest_member_index)."""
    cen = model.centroid_scores(members_pos, members_w, weights)
    S = model.sim_to(members_pos, weights)
    m = min(m_knn, S.shape[1])
    knn = np.partition(S, -m, axis=1)[:, -m:].mean(axis=1)
    return blend * cen + (1 - blend) * knn, S.argmax(axis=1)


def member_loo_scores(sub_model, members_w, m_knn=5, blend=0.0, weights=None):
    """Score of every member against its own profile with the member itself left out. Used to calibrate
    the per-assignment confidence and the 'fits no profile' threshold."""
    cen = sub_model.loo_centroid_scores(members_w, weights)
    S = sub_model.sim_to(np.arange(sub_model.n), weights)
    np.fill_diagonal(S, -np.inf)
    m = min(m_knn, S.shape[1] - 1)
    knn = np.partition(S, -m, axis=1)[:, -m:].mean(axis=1) if m > 0 else cen
    return blend * cen + (1 - blend) * knn


def confidence_from(loo_sorted: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Share of the profile's own members that score no higher than this business (0..1)."""
    return np.searchsorted(loo_sorted, score, side="right") / max(len(loo_sorted), 1)


def reason_text(row_feats: list[str], sim: float, nearest_name: str | None,
                missing: list[str] | None = None, n_total: int = 0, own: str | None = None) -> str:
    """Deterministic Greek reason: shared significant features (k of N), where it differs, similarity, nearest client."""
    bits = []
    if own:
        bits.append(own)
    if row_feats:
        head = f"Κοινά χαρακτηριστικά με το προφίλ ({len(row_feats)}/{n_total})" if n_total else "Κοινά χαρακτηριστικά με το προφίλ"
        bits.append(head + ": " + "; ".join(row_feats))
    if missing:
        bits.append("διαφέρει σε: " + "; ".join(missing))
    bits.append(f"ομοιότητα με τους πιο κοντινούς πελάτες {sim:.0%}")    # one ingredient of the ranking, not the ranking itself
    if nearest_name:
        bits.append(f"πιο κοντινός πελάτης: {nearest_name}")
    return " · ".join(bits)
