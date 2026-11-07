"""Activity Clarity Index (Δείκτης καθαρότητας δραστηριότητας), 0-100 per business.

How "tidy" a business' declared activity profile is: one clear main KAD, without a long tail of unrelated declared codes.
KADs are self-declared and noisy (57% of the non-main codes in the data are outside food service), so this flags
businesses whose activity data should be read with care. It describes the DATA QUALITY of the declaration only - it is not a
judgement of the business, and it does not change the ranking.

For every declared non-main KAD we measure how related it is to the main KAD:
  relevance = 1 if it belongs to the same 2-digit division (e.g. 56 = food service), otherwise its semantic closeness to the main
  code (sigmoid of the description-embedding similarity compared with a typical pair of codes, 0..1).
  unrelated = sum over non-main codes of (1 - relevance)
  index = 100 / (1 + unrelated / 3)            0 unrelated -> 100, 3 -> 50, 9 -> 25
  (x 0.85 when no main KAD is declared at all)
Levels: >= 80 "Καθαρό", 50-79 "Μέτριο", < 50 "Θορυβώδες"."""
from __future__ import annotations
import numpy as np
import pandas as pd

HALF = 3.0          # number of fully unrelated codes that halves the index
LEVELS = ((80, "Καθαρό"), (50, "Μέτριο"), (0, "Θορυβώδες"))


def level(x: float) -> str:
    return next(name for lo, name in LEVELS if x >= lo)


def compute(activity: pd.DataFrame, company_ids: pd.Series, kad_ids: list, emb) -> pd.DataFrame:
    a = activity[["company_id", "activity_id", "atype"]].dropna().drop_duplicates(["company_id", "activity_id"]).copy()
    a["activity_id"] = a["activity_id"].astype(str)
    a = a[a["company_id"].isin(set(company_ids))]
    kidx = {k: i for i, k in enumerate(map(str, kad_ids))}
    E = np.asarray(emb, np.float32)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(0)
    ii, jj = rng.integers(0, len(E), 20000), rng.integers(0, len(E), 20000)
    off = (E[ii] * E[jj]).sum(1)[ii != jj]
    mu, sd = float(off.mean()), float(off.std() + 1e-6)

    prim = a[a["atype"] == "primary"].drop_duplicates("company_id").set_index("company_id")["activity_id"]
    oth = a[a["atype"] != "primary"].copy()
    oth["main"] = oth["company_id"].map(prim)
    has_main = oth["main"].notna().to_numpy()
    same_div = np.zeros(len(oth), bool)
    same_div[has_main] = oth["activity_id"].str[:2].to_numpy()[has_main] == oth["main"].str[:2].to_numpy()[has_main]
    sim = np.full(len(oth), mu, np.float32)
    ci = oth["activity_id"].map(kidx).to_numpy(dtype=float)
    mi = oth["main"].map(kidx).to_numpy(dtype=float)
    ok = has_main & ~np.isnan(ci) & ~np.isnan(mi)
    sim[ok] = (E[ci[ok].astype(int)] * E[mi[ok].astype(int)]).sum(1)
    plaus = 1 / (1 + np.exp(-2 * (sim - mu) / sd))
    rel = np.where(same_div, 1.0, plaus)
    rel[~has_main] = 0.5                      # no main code to compare with: neutral
    oth["unrel"] = 1 - rel
    oth["related"] = rel >= 0.5
    g = oth.groupby("company_id").agg(n_other=("activity_id", "size"), unrelated=("unrel", "sum"), n_related=("related", "sum"))
    out = pd.DataFrame(index=pd.Index(company_ids.values, name="company_id"))
    out = out.join(g).fillna({"n_other": 0, "unrelated": 0.0, "n_related": 0})
    out["has_main"] = out.index.isin(prim.index)
    idx = 100.0 / (1.0 + out["unrelated"] / HALF) * np.where(out["has_main"], 1.0, 0.85)
    out["clarity"] = idx.round(0).astype(int)
    out["n_other"] = out["n_other"].astype(int)
    out["n_unrelated"] = (out["n_other"] - out["n_related"]).astype(int)
    out["level"] = [level(x) for x in out["clarity"]]
    res = out[["clarity", "level", "n_other", "n_unrelated", "has_main"]].copy()
    oth["relevance"] = rel
    res.attrs["codes"] = oth[["company_id", "activity_id", "atype", "relevance"]].reset_index(drop=True)   # per declared code, for the detail view
    return res


def explain(row) -> str:
    if not row["has_main"]:
        return "Δεν έχει δηλωθεί κύριος ΚΑΔ."
    if row["n_other"] == 0:
        return "Μόνο κύριος ΚΑΔ, χωρίς άλλους δηλωμένους κωδικούς."
    return (f"{row['n_other']} δηλωμένοι ΚΑΔ εκτός του κύριου· {row['n_unrelated']} από αυτούς άσχετοι με την κύρια δραστηριότητα.")


def explain_all(clar: pd.DataFrame) -> list[str]:
    """explain() for every row at once (vectorised; iterating a 24k-row DataFrame row by row took ~45 s)."""
    has, n_o, n_u = clar["has_main"].to_numpy(), clar["n_other"].to_numpy(), clar["n_unrelated"].to_numpy()
    return [explain({"has_main": bool(h), "n_other": int(o), "n_unrelated": int(u)}) for h, o, u in zip(has, n_o, n_u)]


def profile_summary(clar: pd.DataFrame, member_pos: np.ndarray) -> dict:
    c = clar["clarity"].to_numpy()
    noisy = clar["level"].to_numpy() == "Θορυβώδες"
    return {"profile_mean": round(float(c[member_pos].mean()), 1), "market_mean": round(float(c.mean()), 1),
            "profile_noisy_share": round(float(noisy[member_pos].mean()), 3), "market_noisy_share": round(float(noisy.mean()), 3)}
