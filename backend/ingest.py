"""Part 1: robust loading of the challenge CSV/Excel files.

Brief facts handled here: CSV is UTF-8, ';' separated, decimal comma, empty values = NULL.
VATs are kept as 9-digit strings (the client Excel contains leading zeros).
"""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

NA = ["NULL", "null", "Null", ""]


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")


def normalize_vat(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, np.integer)) or (isinstance(x, float) and float(x).is_integer()):
        x = str(int(x))
    t = str(x).strip()
    t = re.sub(r"[.,]0+$", "", t)          # "123456789.0" (a number read from Excel/CSV) -> "123456789"
    s = re.sub(r"\D", "", t)
    if not s:
        return None
    return s.zfill(9) if len(s) <= 9 else s


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, na_values=NA, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")


def _parse_dates(s: pd.Series) -> pd.Series:
    """Try several formats, keep whichever parses most values. Parsed on the DISTINCT strings (there are only ~2,000 days
    in the data, not 64,000 rows) and mapped back: same result, ~20x less work."""
    counts = s.value_counts()
    uniq = pd.Series(counts.index, index=counts.index)
    best, best_rate = None, -1.0
    for fmt in (None, "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            p = pd.to_datetime(uniq, format=fmt, errors="coerce")
        except Exception:
            continue
        rate = float((p.notna() * counts.reindex(p.index)).sum()) / max(int(s.notna().sum()), 1) if len(counts) else 0.0
        if rate > best_rate:
            best, best_rate = p, rate
        if best_rate >= 0.999:
            break
    if best is None:
        return pd.to_datetime(s, errors="coerce")
    return s.map(dict(zip(best.index, best.values))).astype("datetime64[ns]")


def find_file(data_dir: Path, name: str) -> Path:
    data_dir = Path(data_dir)
    hits = list(data_dir.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {data_dir}. Run `python run.py unlock` first.")
    return hits[0]


def atype_norm(s: pd.Series) -> pd.Series:
    s = s.fillna("")
    lut = {u: strip_accents(u).lower().strip() for u in s.unique()}          # a handful of distinct values, not 64,000 rows
    t = s.map(lut)
    out = pd.Series("secondary", index=s.index)
    out[t.str.startswith("κυρ") | t.str.startswith("prim") | t.str.startswith("main")] = "primary"
    out[t.str.startswith("βοηθ") | t.str.startswith("aux")] = "auxiliary"
    return out


def load_universe(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    comp = _read_csv(find_file(data_dir, "vwHackCompany.csv"))
    comp["vat"] = comp["vat"].map(normalize_vat)
    comp["is_branch"] = pd.to_numeric(comp["is_branch"], errors="coerce").fillna(0).astype(int)
    comp["latitude"] = _num(comp["latitude"])
    comp["longitude"] = _num(comp["longitude"])
    comp["incorporation_date"] = _parse_dates(comp["incorporation_date"])
    comp = comp.sort_values("company_id", kind="stable").reset_index(drop=True)

    act = _read_csv(find_file(data_dir, "vwHackCompanyActivity.csv"))
    act["atype"] = atype_norm(act["activity_type"])
    act["date_from"] = _parse_dates(act["date_from"]) if "date_from" in act else pd.NaT

    kad = _read_csv(find_file(data_dir, "vwHackActivities.csv"))
    kad_desc = dict(zip(kad["activity_id"], kad["description"].fillna("")))
    return {"companies": comp, "activity": act, "kad_desc": kad_desc}


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, dtype=str)
    head = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[:1] or [""]
    sep = max((";", ",", "\t"), key=head[0].count)
    return pd.read_csv(path, dtype=str, sep=sep, encoding="utf-8-sig")


def load_clients(path: Path) -> pd.DataFrame:
    """Client list from .xlsx or .csv/.tsv. Needs a VAT (ΑΦΜ) column; turnover (τζίρος) is optional."""
    raw = _read_table(path)
    if raw.shape[1] == 0 or len(raw) == 0:
        raise ValueError("Το αρχείο είναι κενό / The file is empty.")
    cols = {c: strip_accents(c).lower() for c in raw.columns}
    vat_col = next((c for c, n in cols.items() if "αφμ" in n or "vat" in n or "afm" in n), None)
    if vat_col is None:                      # no recognisable header: accept the first column if it looks like VATs
        first = raw.iloc[:, 0].astype(str).str.replace(r"\D", "", regex=True)
        if (first.str.len().between(8, 9)).mean() < 0.8:
            raise ValueError("Δεν βρέθηκε στήλη ΑΦΜ / No VAT (ΑΦΜ) column found. Expected columns: ΑΦΜ, Τζίρος (€).")
        vat_col = raw.columns[0]
    rev_col = next((c for c, n in cols.items() if "τζιρ" in n or "revenue" in n or "turnover" in n), None)
    out = pd.DataFrame({"vat": raw[vat_col].map(normalize_vat)})
    if rev_col is not None:
        out["revenue"] = pd.to_numeric(raw[rev_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    else:  # revenue is optional
        out["revenue"] = np.nan
    out = out.dropna(subset=["vat"]).drop_duplicates("vat").reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("Δεν βρέθηκαν έγκυροι ΑΦΜ / No valid VAT numbers found.")
    return out


def match_clients(companies: pd.DataFrame, clients: pd.DataFrame, mode: str = "split") -> tuple[pd.DataFrame, dict]:
    """Map client VATs to shops.

    mode='split'  : revenue of a VAT is split equally across its shops (default, no double counting)
    mode='head'   : only the head office (is_branch == 0, else first shop) represents the VAT
    mode='full'   : every shop gets the full revenue (double counts; for ablation only)
    Returns (client_shops, stats). client_shops has positional index `pos` into `companies`.
    """
    c = companies[["company_id", "vat", "is_branch"]].copy()
    c["pos"] = np.arange(len(c))
    m = c.merge(clients, on="vat", how="inner")
    n_shops = m.groupby("vat")["pos"].transform("size")
    if mode == "head":
        m = m.sort_values(["vat", "is_branch", "pos"]).drop_duplicates("vat", keep="first")
        m["weight"] = m["revenue"]
    elif mode == "full":
        m["weight"] = m["revenue"]
    else:
        m["weight"] = m["revenue"] / n_shops
    m["weight"] = m["weight"].fillna(m["weight"].median() if m["weight"].notna().any() else 1.0)
    m = m.reset_index(drop=True)
    stats = {
        "client_vats": int(len(clients)),
        "matched_vats": int(m["vat"].nunique()),
        "unmatched_vats": int(len(clients) - m["vat"].nunique()),
        "client_shops": int(len(m)),
        "mode": mode,
    }
    return m[["pos", "company_id", "vat", "revenue", "weight"]], stats
