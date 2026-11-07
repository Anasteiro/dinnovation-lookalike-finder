"""What-if: change how much each kind of information counts and see how the lookalike ranking moves.
Sliders are multipliers (1.0 = the product's setting) for four groups of information: activity (KAD), location, legal form, age.
At 1.0/1.0/1.0/1.0 the result is exactly the product's ranking (tested)."""
from __future__ import annotations
import numpy as np
from . import config as C, lookalike as L, rankers as R

GROUPS = ("activity", "geo", "legal", "age")
LABELS = {"activity": "Δραστηριότητα (ΚΑΔ)", "geo": "Τοποθεσία", "legal": "Νομική μορφή", "age": "Ηλικία επιχείρησης"}
BASE_PU = {"activity": .55, "geo": .25, "legal": .12, "age": .08}       # block weights of the PU features (see rankers.dense_features_v2)


def clean(mult: dict | None) -> dict:
    m = {g: 1.0 for g in GROUPS}
    for g, v in (mult or {}).items():
        if g in m:
            m[g] = float(min(2.0, max(0.0, v)))
    if sum(m.values()) <= 0:
        m["activity"] = 1.0
    return m


def block_weights(mult: dict) -> dict:
    """Similarity block weights (admin/struct keep their defaults; 'geo' scales both the admin and the geography block)."""
    w = dict(C.DEFAULT_BLOCK_WEIGHTS)
    w["activity"] *= mult["activity"]
    w["geo"] *= mult["geo"]
    w["admin"] *= mult["geo"]
    w["legal"] *= mult["legal"]
    w["age"] *= mult["age"]
    return w


def pu_features(ctx, mult: dict):
    bw = {g: BASE_PU[g] * mult[g] for g in GROUPS}
    blocks = tuple(b for b, g in (("activity", "activity"), ("legal", "legal"), ("age", "age")) if mult[g] > 0) + ("struct",)
    return R.dense_features_v2(ctx.model, ctx.companies["latitude"], ctx.companies["longitude"],
                               blocks=blocks, bw={**bw, "struct": .1}, geo=mult["geo"] > 0)


def rank_scores(ctx, state: dict, mult: dict, Z=None) -> np.ndarray:
    """n x k ranking scores (same recipe as analysis.run_analysis) under the given multipliers."""
    wts = block_weights(mult)
    Zp = pu_features(ctx, mult) if Z is None else Z
    out = np.zeros((len(ctx.companies), len(state["members"])))
    for c, (mpos, mw) in enumerate(state["members"]):
        sc, _ = L.profile_scores(ctx.model, mpos, mw, C.KNN_M, C.CENTROID_BLEND, wts)
        out[:, c] = C.RANK_W_KNN * R._pct(sc) + C.RANK_W_PU * R._pct(R.pu_scores(Zp, mpos, mw / mw.mean()))
    return out


def global_order(rs: np.ndarray, share: np.ndarray, cand: np.ndarray):
    g = R.merge_proportional(rs[cand], share)
    return cand[np.argsort(-g, kind="stable")]


def run(ctx, state: dict, mult: dict | None, top: int = 25) -> dict:
    mult = clean(mult)
    share = state["share"]
    cand = state["cand"]
    base = state.get("_default_order")
    if base is None:
        base = state["_default_order"] = global_order(rank_scores(ctx, state, clean(None)), share, cand)
    order = base if mult == clean(None) else global_order(rank_scores(ctx, state, mult), share, cand)
    pos_base = np.empty(len(ctx.companies), int); pos_base[base] = np.arange(len(base))
    comp = ctx.companies
    items = []
    for r, i in enumerate(order[:top], start=1):
        items.append({"rank": r, "company_id": str(comp["company_id"].iat[i]), "name": str(comp["company_name_el"].iat[i] or ""),
                      "municipality": str(comp["municipality_name"].iat[i] or ""), "default_rank": int(pos_base[i]) + 1,
                      "moved": int(pos_base[i]) + 1 - r})
    ov = {str(k): int(len(set(order[:k]) & set(base[:k]))) for k in C.KS}
    return {"multipliers": mult, "labels": LABELS, "items": items, "overlap_with_default": ov, "ks": list(C.KS),
            "is_default": mult == clean(None)}
