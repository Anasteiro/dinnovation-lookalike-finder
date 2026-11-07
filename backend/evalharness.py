"""Part 2: the evaluation harness. Mirrors the organisers' hold-out test (brief section 8.1):
hide ~20% of the clients, rank the universe (minus the visible clients), measure Recall@k, and
compare with the same-main-KAD baseline. Splits are by VAT so shops of one VAT never leak."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C


def rank_order(scores, exclude_idx, rng) -> np.ndarray:
    s = np.asarray(scores, dtype=float).copy()
    s[exclude_idx] = -np.inf
    tie = rng.random(len(s))                       # random tie-break (many ties in the baseline)
    return np.lexsort((tie, -s))


def baseline_scorer(primary_kad_arr: np.ndarray):
    """Same main KAD as the clients, ordered by that KAD's frequency in the client list."""
    pk = pd.Series(primary_kad_arr)
    def fn(train_idx, train_w):
        freq = pk.iloc[train_idx].value_counts()
        return pk.map(freq).fillna(0).to_numpy(float)
    return fn


def cross_validate(n_universe, client_pos, client_w, groups, scorers: dict, repeats=30,
                   test_frac=0.2, ks=C.KS, seed=C.SEED) -> pd.DataFrame:
    client_pos, client_w, groups = np.asarray(client_pos), np.asarray(client_w, float), np.asarray(groups)
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(repeats):
        n_test = max(1, int(round(test_frac * len(ug))))
        test_g = ug[rng.permutation(len(ug))[:n_test]]
        is_test = np.isin(groups, test_g)
        train, test = client_pos[~is_test], client_pos[is_test]
        for name, fn in scorers.items():
            scores = fn(train, client_w[~is_test])
            order = rank_order(scores, train, np.random.default_rng(seed * 1000 + r))
            rank = np.empty(n_universe, dtype=np.int64)
            rank[order] = np.arange(n_universe)
            tr = rank[test]
            row = {"repeat": r, "name": name, "median_rank_pct": float(np.median(tr) / n_universe)}
            for k in ks:
                row[f"recall@{k}"] = float((tr < k).mean())
            # VAT-level recall: a hidden VAT counts as found if any of its shops is in the top-k
            tg = groups[is_test]
            for k in ks:
                hit = pd.Series(tr < k).groupby(tg).any()
                row[f"vat_recall@{k}"] = float(hit.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, baseline="baseline", ks=C.KS) -> pd.DataFrame:
    out = []
    base = df[df["name"] == baseline].set_index("repeat")
    R = df["repeat"].nunique()
    for name, g in df.groupby("name", sort=False):
        g = g.set_index("repeat")
        row = {"model": name}
        for k in ks:
            col = f"recall@{k}"
            row[col] = g[col].mean()
            row[col + "_ci95"] = 1.96 * g[col].std(ddof=1) / np.sqrt(R) if R > 1 else np.nan
            if baseline in set(df["name"]) and name != baseline:
                d = g[col] - base[col]
                row[f"lift_vs_base@{k}"] = d.mean()
                row[f"paired_wins@{k}"] = float((d > 0).mean())
        row["median_rank_pct"] = g["median_rank_pct"].mean()
        out.append(row)
    return pd.DataFrame(out)
