"""In-app accuracy check ("Επαλήθευσε"): hide 20% of the uploaded clients (split by VAT), rebuild the profiles and ranking from
the other 80% with exactly the product's method, and measure where the hidden clients land. Compared with the brief's
baseline (same primary KAD) and with plain kNN. Repeated random splits -> mean and 95% CI."""
from __future__ import annotations
import time
import numpy as np
from . import config as C, evalharness as H, rankers as R

NOTE = ("Οι υπολογισμοί γίνονται στους δικούς σας πελάτες: κρύβουμε τυχαία το 20% (ανά ΑΦΜ), ξαναχτίζουμε προφίλ και κατάταξη από το υπόλοιπο 80% "
        "και μετράμε πόσοι από τους κρυμμένους βρίσκονται στις κορυφαίες k προτάσεις. Η ρύθμιση των βαρών έγινε στους ίδιους πελάτες, "
        "οπότε το αποτέλεσμα στο κρυφό σύνολο του διαγωνισμού μπορεί να είναι κάπως χαμηλότερο.")


def run_benchmark(ctx, k: int = 2, repeats: int = 8, seed: int = 321, progress=lambda i, n: None, mult: dict | None = None) -> dict:
    t0 = time.time()
    s = ctx.client_shops
    w = R.revenue_weights(s["weight"].to_numpy(float), "log")
    wts = None
    if mult is None:
        Z = R.pu_features(ctx.model, ctx.companies["latitude"], ctx.companies["longitude"])
    else:                                   # what-if multipliers (see whatif.py)
        from . import whatif as W
        mult = W.clean(mult); wts = W.block_weights(mult); Z = W.pu_features(ctx, mult)
    scorers = {"baseline": H.baseline_scorer(ctx.primary_kad),
               "knn": R.knn_scorer(ctx.model, C.KNN_M, 0.0, wts),
               "product": R.profile_pu_scorer(ctx.model, Z, k=max(1, k), w_knn=C.RANK_W_KNN, w_pu=C.RANK_W_PU, w_puall=0.0, weights=wts)}
    df = H.cross_validate(len(ctx.companies), s["pos"].to_numpy(), w, s["vat"].to_numpy(), scorers, repeats=repeats, seed=seed)
    summ = H.summarize(df).set_index("model")
    labels = {"baseline": "Βάση αναφοράς (ίδιος κύριος ΚΑΔ)", "knn": "Μόνο κοντινότεροι πελάτες (kNN)", "product": "Η μέθοδός μας (προφίλ + PU-learning)"}
    methods = {}
    for name in scorers:
        r = summ.loc[name]
        m = {"label": labels[name]}
        for kk in C.KS:
            m[f"recall@{kk}"] = round(float(r[f"recall@{kk}"]), 4)
            h = r[f"recall@{kk}_ci95"]                       # half-width -> [lower, upper] interval, which is what the UI draws
            if not np.isnan(h):
                m[f"recall@{kk}_ci95"] = [round(max(0.0, float(r[f"recall@{kk}"] - h)), 4), round(min(1.0, float(r[f"recall@{kk}"] + h)), 4)]
        m["median_rank_pct"] = round(float(r["median_rank_pct"]), 4)
        methods[name] = m
    b, p = summ.loc["baseline"], summ.loc["product"]
    n_vat = int(len(np.unique(s["vat"].to_numpy())))
    return {"status": "done", "repeats": repeats, "n_client_vats": n_vat, "n_hidden_per_split": int(round(0.2 * n_vat)),
            "universe": int(len(ctx.companies)), "ks": list(C.KS), "methods": methods,
            "lift_vs_baseline": {str(kk): round(float(p[f"recall@{kk}"] / max(b[f"recall@{kk}"], 1e-9)), 2) for kk in C.KS},
            "paired_wins": round(float(np.mean([p[f"paired_wins@{kk}"] for kk in C.KS])), 3),
            "paired_wins_by_k": {str(kk): round(float(p[f"paired_wins@{kk}"]), 2) for kk in C.KS},
            "random_level": {str(kk): round(kk / len(ctx.companies), 4) for kk in C.KS},
            "seconds": round(time.time() - t0, 1), "note": NOTE}
