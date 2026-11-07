"""Part 3 (first half): business representation = several L2-normalised BLOCKS.

Similarity between two businesses = weighted average of per-block cosines. Because blocks are kept
separate, block weights can be tuned by cross-validation without recomputing anything, and the
SAME metric forms profiles and assigns businesses to them (requirement A3).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
from . import config as C


def _l2_rows(X):
    if sp.issparse(X):
        n = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        return sp.diags(1.0 / np.maximum(n, 1e-12)) @ X
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


# ----------------------------------------------------------------------------- activity (KAD)
def build_kad_weights(activity: pd.DataFrame, company_ids: pd.Series, kad_ids: list[str],
                      primary_w=C.PRIMARY_WEIGHT, other_total=C.OTHER_TOTAL, aux_factor=C.AUX_FACTOR,
                      drop_other=None, emb=None) -> sp.csr_matrix:
    """Sparse (n_companies x n_kad) weights. Primary KADs carry `primary_w`; every other declared
    KAD shares a fixed total `other_total` (auxiliary counted at `aux_factor`). `drop_other` is an
    optional callable(activity_id Series)->bool mask of declarative-noise codes to ignore."""
    rank = {"primary": 0, "secondary": 1, "auxiliary": 2}
    a = activity[["company_id", "activity_id", "atype"]].dropna().copy()
    a["r"] = a["atype"].map(rank)
    a = a.sort_values("r", kind="stable").drop_duplicates(["company_id", "activity_id"])
    if drop_other is not None:
        a = a[(a["atype"] == "primary") | ~drop_other(a["activity_id"])]
    cidx = pd.Series(np.arange(len(company_ids)), index=company_ids.values)
    kidx = pd.Series(np.arange(len(kad_ids)), index=kad_ids)
    a = a[a["company_id"].isin(cidx.index) & a["activity_id"].isin(kidx.index)].copy()
    a["row"] = a["company_id"].map(cidx).values
    a["col"] = a["activity_id"].map(kidx).values
    is_p = (a["atype"] == "primary").values
    a["f"] = np.where(is_p, 1.0, np.where(a["atype"] == "auxiliary", aux_factor, 1.0))
    n_p = a[is_p].groupby("row").size().reindex(range(len(company_ids)), fill_value=0)
    f_other = a[~is_p].groupby("row")["f"].sum().reindex(range(len(company_ids)), fill_value=0.0)
    n_p_row = a["row"].map(n_p).values
    tot_other = np.where(n_p_row > 0, other_total, primary_w)       # no primary declared -> others carry 1.0
    w = np.where(is_p, primary_w / np.maximum(n_p_row, 1),
                 tot_other * a["f"].values / np.maximum(a["row"].map(f_other).values, 1e-12))
    if emb is not None:
        # Noise-aware damping (idea from the second developer): a declared secondary code counts only as much as it is
        # semantically plausible next to the company's primary code (sigmoid of how much closer than a typical pair of codes);
        # the total secondary mass is capped, so firms with few/implausible extra codes are not inflated to the full mass.
        E = np.asarray(emb, np.float32); E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        rng = np.random.default_rng(0); ii = rng.integers(0, len(E), 20000); jj = rng.integers(0, len(E), 20000)
        off = (E[ii] * E[jj]).sum(1)[ii != jj]; mu, sd = off.mean(), off.std() + 1e-6
        prim_col = pd.Series(a.loc[is_p].drop_duplicates("row").set_index("row")["col"])
        pc = a["row"].map(prim_col).to_numpy(dtype=float)
        ok = ~np.isnan(pc)
        sim = np.zeros(len(a), np.float32)
        sim[ok] = (E[a["col"].values[ok]] * E[pc[ok].astype(int)]).sum(1)
        plaus = 1 / (1 + np.exp(-2 * (sim - mu) / sd))
        raw = np.where(is_p, 0.0, a["f"].values * (0.25 + 0.75 * plaus))
        a["raw"] = raw
        tot = a["row"].map(a[~is_p].groupby("row")["raw"].sum()).fillna(0).to_numpy()
        scale = np.minimum(1.0, tot_other / np.maximum(tot, 1e-12))
        w = np.where(is_p, primary_w / np.maximum(n_p_row, 1), raw * scale)
    return sp.csr_matrix((w, (a["row"].values, a["col"].values)), shape=(len(company_ids), len(kad_ids)))


def primary_kad(activity: pd.DataFrame, company_ids: pd.Series) -> np.ndarray:
    p = activity[activity["atype"] == "primary"].drop_duplicates("company_id").set_index("company_id")["activity_id"]
    return company_ids.map(p).fillna("NA").values


def activity_block(W: sp.csr_matrix, E: np.ndarray) -> np.ndarray:
    return _l2_rows(np.asarray(W @ E, dtype=np.float32))


# ----------------------------------------------------------------------------- geography
def geo_block(lat, lon, municipality, scales=(0.02, 0.1, 0.5), scale_w=(1.0, 1.0, 1.0)) -> sp.csr_matrix:
    """Multi-scale grid one-hots (~2 km, ~11 km, ~55 km). Missing GPS falls back to the municipality
    centroid; still missing -> zero row. Similarity = weighted fraction of shared cells."""
    df = pd.DataFrame({"lat": lat, "lon": lon, "m": municipality})
    cen = df.groupby("m")[["lat", "lon"]].transform("mean")
    lat_f = df["lat"].fillna(cen["lat"]).to_numpy(float)
    lon_f = df["lon"].fillna(cen["lon"]).to_numpy(float)
    ok = ~(np.isnan(lat_f) | np.isnan(lon_f))
    rows, cols, vals, off = [], [], [], 0
    norm = np.sqrt(sum(w * w for w in scale_w))
    for s, w in zip(scales, scale_w):
        key = pd.Series(np.where(ok, np.floor(lat_f / s).astype("int64").astype(str) + ":" +
                                 np.floor(lon_f / s).astype("int64").astype(str), None))
        codes, uniq = pd.factorize(key[ok])
        rows.append(np.flatnonzero(ok)); cols.append(codes + off); vals.append(np.full(ok.sum(), w / norm))
        off += len(uniq)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(df), off))


def onehot_block(series: pd.Series, weight=1.0) -> sp.csr_matrix:
    s = series.fillna("UNK").astype(str)
    codes, uniq = pd.factorize(s)
    return sp.csr_matrix((np.full(len(s), float(weight)), (np.arange(len(s)), codes)), shape=(len(s), len(uniq)))


def admin_block(prefecture: pd.Series, municipality: pd.Series) -> sp.csr_matrix:
    X = sp.hstack([onehot_block(prefecture, 0.5), onehot_block(municipality, 1.0)]).tocsr()
    return _l2_rows(X)


def age_block(inc_date: pd.Series, n_centers=8) -> np.ndarray:
    age = ((C.REF_DATE - inc_date).dt.days / 365.25).to_numpy(float)
    ok = ~np.isnan(age)
    lo, hi = np.nanmin(age), np.nanmax(age)
    centers = np.linspace(lo, hi, n_centers)
    sigma = max((hi - lo) / (n_centers - 1), 0.25)
    X = np.exp(-((np.nan_to_num(age, nan=lo)[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2))
    X[~ok] = 0.0
    return _l2_rows(X.astype(np.float32))


def struct_block(is_branch: pd.Series, vat: pd.Series) -> sp.csr_matrix:
    n = vat.map(vat.value_counts())
    bucket = pd.cut(n, [0, 1, 2, 5, 10**9], labels=["1", "2", "3-5", "6+"]).astype(str)
    return _l2_rows(onehot_block(is_branch.astype(str) + "|" + bucket))


# ----------------------------------------------------------------------------- block model
class BlockModel:
    """Holds L2-normalised block matrices and computes weighted-cosine similarities."""

    def __init__(self, blocks: dict, weights: dict | None = None):
        self.blocks = blocks
        self.weights = dict(weights or C.DEFAULT_BLOCK_WEIGHTS)
        self.n = next(iter(blocks.values())).shape[0]

    def _w(self, weights):
        w = dict(self.weights if weights is None else weights)
        w = {k: v for k, v in w.items() if k in self.blocks and v > 0}
        return w, sum(w.values())

    def sim_to(self, idx, weights=None) -> np.ndarray:
        """(n x len(idx)) weighted-average cosine of every business to the businesses `idx`."""
        w, tot = self._w(weights)
        S = np.zeros((self.n, len(idx)), dtype=np.float32)
        for b, wb in w.items():
            X = self.blocks[b]
            part = X @ X[idx].T
            S += wb * (part.toarray() if sp.issparse(part) else part).astype(np.float32)
        return S / tot

    def centroid_scores(self, idx, cw, weights=None) -> np.ndarray:
        w, tot = self._w(weights)
        cw = np.asarray(cw, float)
        s = np.zeros(self.n, dtype=np.float32)
        for b, wb in w.items():
            X = self.blocks[b]
            c = X[idx].T @ cw
            c = np.asarray(c).ravel()
            c = c / max(np.linalg.norm(c), 1e-12)
            s += wb * np.asarray(X @ c).ravel().astype(np.float32)
        return s / tot

    def knn_scores(self, idx, cw, m=5, alpha=0.0, weights=None) -> np.ndarray:
        S = self.sim_to(idx, weights)
        if alpha:
            cw = np.asarray(cw, float)
            S = S * ((cw / cw.max()) ** alpha)[None, :]
        m = min(m, S.shape[1])
        return np.partition(S, -m, axis=1)[:, -m:].mean(axis=1)

    def subset(self, idx) -> "BlockModel":
        """Model restricted to the rows `idx` (used for clustering the clients only)."""
        return BlockModel({k: X[idx] for k, X in self.blocks.items()}, self.weights)

    def loo_centroid_scores(self, cw, weights=None) -> np.ndarray:
        """Leave-one-out centroid score of every row of THIS (sub)model against the weighted centroid
        of all the other rows. Closed form per block, no loops over members."""
        w, tot = self._w(weights)
        cw = np.asarray(cw, float)
        out = np.zeros(self.n, dtype=np.float64)
        for b, wb in w.items():
            X = self.blocks[b]
            sv = np.asarray(X.T @ cw).ravel()
            a = np.asarray(X @ sv).ravel()
            xx = np.asarray(X.multiply(X).sum(axis=1)).ravel() if sp.issparse(X) else (X * X).sum(axis=1)
            norm2 = float(sv @ sv) - 2 * cw * a + cw ** 2 * xx
            out += wb * (a - cw * xx) / np.sqrt(np.maximum(norm2, 1e-12))
        return out / tot
