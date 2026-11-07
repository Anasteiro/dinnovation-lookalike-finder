"""Part 4b: statistical evidence per profile (requirement A4).

One-sided Fisher exact test (= hypergeometric survival function) of every categorical feature in the
profile vs the NON-CLIENT universe, Benjamini-Hochberg correction across all tested features of that
profile, a minimum support, and a shrunk lift so that rare features cannot look impressive by luck.
"""
from __future__ import annotations
import numpy as np
from .labels import short_kad
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

MONTHS_NOTE = "features: primary KAD, KAD group, declared extra KAD, municipality, prefecture, legal form, founding year, branch"


def build_feature_matrix(comp: pd.DataFrame, act: pd.DataFrame, desc: dict):
    """Binary sparse (n_companies x n_features) + readable Greek labels."""
    n = len(comp)
    parts, labels = [], {}

    def add(rows, names, fmt):
        d = pd.DataFrame({"r": rows, "f": names}).dropna()
        d = d.drop_duplicates()
        parts.append(d)
        for f in d["f"].unique():
            labels[f] = fmt(f)

    cid = pd.Series(np.arange(n), index=comp["company_id"].values)
    a = act[["company_id", "activity_id", "atype"]].dropna()
    a = a[a["company_id"].isin(cid.index)]
    a_row = a["company_id"].map(cid).to_numpy()
    prim = (a["atype"] == "primary").to_numpy()
    ids = a["activity_id"].astype(str)

    def kad_label(f):
        code = f.split(":", 1)[1]
        return f"{desc.get(code, code)[:60]} ({code})"

    add(a_row[prim], "kad:" + ids[prim], lambda f: "Κύριος ΚΑΔ: " + kad_label(f))
    def group_label(f):
        g = f.split(":", 1)[1]
        d = desc.get(g + "0000") or next((v for k, v in sorted(desc.items()) if str(k).startswith(g)), None)
        return f"Ομάδα ΚΑΔ {g}" + (f" ({short_kad(d)})" if d else "")
    add(a_row[prim], "kad4:" + ids[prim].str[:4], group_label)
    add(a_row[~prim], "kadx:" + ids[~prim], lambda f: "Δηλωμένος δευτερεύων ΚΑΔ: " + kad_label(f))
    rows = np.arange(n)
    add(rows, "muni:" + comp["municipality_name"].astype("object"), lambda f: "Δήμος " + f.split(":", 1)[1])
    add(rows, "pref:" + comp["prefecture_name"].astype("object"), lambda f: "Νομός " + f.split(":", 1)[1])
    add(rows, "legal:" + comp["legal_type"].astype("object"), lambda f: "Νομική μορφή " + f.split(":", 1)[1])
    yr = comp["incorporation_date"].dt.year
    add(rows, "year:" + yr.astype("Int64").astype(str).where(yr.notna()), lambda f: "Ίδρυση " + f.split(":", 1)[1])
    add(rows, "branch:" + comp["is_branch"].astype(str), lambda f: "Υποκατάστημα" if f.endswith("1") else "Έδρα")
    d = pd.concat(parts, ignore_index=True)
    codes, uniq = pd.factorize(d["f"])
    F = sp.csr_matrix((np.ones(len(d), dtype=np.float32), (d["r"].to_numpy(), codes)), shape=(n, len(uniq)))
    F.data[:] = 1.0
    names = np.asarray(uniq)
    return F, names, labels


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def enrichment(F, names, labels, member_pos, background_pos, min_support=3, shrink_k=5.0, alpha=0.05) -> pd.DataFrame:
    member_pos = np.asarray(member_pos)
    n_mem, n_bg = len(member_pos), len(background_pos)
    a = np.asarray(F[member_pos].sum(axis=0)).ravel()
    c = np.asarray(F[background_pos].sum(axis=0)).ravel()
    M, Nf = n_mem + n_bg, a + c
    p = np.ones(len(a))
    tested = a >= min_support
    p[tested] = hypergeom.sf(a[tested] - 1, M, Nf[tested], n_mem)
    q = np.ones(len(a))
    q[tested] = benjamini_hochberg(p[tested])
    p_bg = np.maximum(c / max(n_bg, 1), 1.0 / (n_bg + 1))
    share = a / max(n_mem, 1)
    lift = share / p_bg
    shr = ((a + shrink_k * p_bg) / (n_mem + shrink_k)) / p_bg
    df = pd.DataFrame({"feature": names, "label": [labels[f] for f in names], "support": a.astype(int),
                       "share_in_profile": share, "share_in_universe": c / max(n_bg, 1),
                       "lift": lift, "shrunk_lift": shr, "p_value": p, "q_value": q})
    df = df[tested & (df["q_value"] < alpha) & (df["shrunk_lift"] > 1.0)]
    return df.sort_values(["q_value", "shrunk_lift"], ascending=[True, False]).reset_index(drop=True)
