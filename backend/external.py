"""External factors per business, joined by municipality (Δήμος). All files are local - nothing is fetched at runtime.

Bundled (backend/assets/external/):
  census2021_municipalities.csv  ELSTAT 2021 census (provisional results) population per municipality, 2021 and 2011.
                                  Source: Hellenic Statistical Authority, machine-readable copy github.com/tdiam/greece-population-census-2021 (CC0).
Derived from the market data itself (no outside source):
  restaurants per 1,000 residents in the municipality (competition / saturation), population change 2011->2021.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from . import ingest as I

ASSETS = Path(__file__).parent / "assets" / "external"
ALIASES = {"ΣΑΛΑΜΙΝΑΣ": "ΣΑΛΑΜΙΝΟΣ", "ΣΤΥΛΙΔΑΣ": "ΣΤΥΛΙΔΟΣ", "ΗΡΩΙΚΗΣΠΟΛΕΩΣΝΑΟΥΣΑΣΗΜΑΘΙΑΣ": "ΗΡΩΙΚΗΣΠΟΛΕΩΣΝΑΟΥΣΑΣ",
           "ΝΕΑΣΦΙΛΑΔΕΛΦΕΙΑΣΝΕΑΣΧΑΛΚΗΔΟΝΑΣ": "ΝΕΑΣΦΙΛΑΔΕΛΦΕΙΑΣΝΕΑΣΧΑΛΚΗΔΟΝΟΣ"}
SOURCE_CENSUS = "ΕΛΣΤΑΤ, Απογραφή 2021 (προσωρινά αποτελέσματα) - αντίγραφο CC0: github.com/tdiam/greece-population-census-2021"


def mkey(s) -> str:
    s = str(s or "").split(" / ")[0].replace("/", "")
    k = re.sub(r"[^Α-ΩA-Z]", "", I.strip_accents(s).upper().replace("&", "ΚΑΙ"))
    return ALIASES.get(k, k)


def _census() -> pd.DataFrame:
    p = pd.read_csv(ASSETS / "census2021_municipalities.csv", dtype={"elstat_municipality_code": str})
    p["key"] = p["municipality_name"].map(mkey)
    return p


def build(companies: pd.DataFrame, data_dir: Path | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (factor table aligned with `companies`, factor catalogue)."""
    comp = companies
    key = comp["municipality_name"].map(mkey)
    pref = comp["prefecture_name"].fillna("").map(lambda s: I.strip_accents(s).upper())
    cen = _census()
    # 'ΗΡΑΚΛΕΙΟΥ' exists twice (Crete and Attica, same code in the source file): the larger one is Heraklion of Crete
    her = cen[cen["key"] == "ΗΡΑΚΛΕΙΟΥ"].sort_values("pop_total_2021")
    single = cen.drop_duplicates("key", keep=False).set_index("key")
    pop21 = key.map(single["pop_total_2021"]).astype(float)
    pop11 = key.map(single["pop_total_2011"]).astype(float)
    if len(her) == 2:
        is_her = key == "ΗΡΑΚΛΕΙΟΥ"
        crete = is_her & pref.str.contains("ΗΡΑΚΛΕΙ")
        attica = is_her & ~crete
        pop21[crete], pop11[crete] = her.iloc[1]["pop_total_2021"], her.iloc[1]["pop_total_2011"]
        pop21[attica], pop11[attica] = her.iloc[0]["pop_total_2021"], her.iloc[0]["pop_total_2011"]
    shops = key.map(key.value_counts())
    F = pd.DataFrame({
        "muni_population": pop21.values,
        "muni_pop_change_pct": ((pop21 - pop11) / pop11 * 100).round(2).values,
        "muni_shops_per_1000": (shops / pop21 * 1000).round(3).values,
    }, index=comp.index)
    cat = [
        {"key": "muni_population", "label": "Πληθυσμός δήμου (2021)", "unit": "κάτοικοι", "source": SOURCE_CENSUS, "higher": "μεγαλύτερος δήμος"},
        {"key": "muni_pop_change_pct", "label": "Μεταβολή πληθυσμού δήμου 2011→2021", "unit": "%", "source": SOURCE_CENSUS, "higher": "δήμος που μεγαλώνει"},
        {"key": "muni_shops_per_1000", "label": "Επιχειρήσεις εστίασης ανά 1.000 κατοίκους", "unit": "ανά 1.000", "source": "Υπολογισμός από τα δεδομένα αγοράς + ΕΛΣΤΑΤ 2021", "higher": "περισσότερος ανταγωνισμός/τουρισμός"},
    ]
    for c in cat:
        v = F[c["key"]]
        c["coverage"] = round(float(v.notna().mean()), 3)
        c["min"] = None if v.notna().sum() == 0 else float(np.nanmin(v))
        c["max"] = None if v.notna().sum() == 0 else float(np.nanmax(v))
        c["market_median"] = None if v.notna().sum() == 0 else float(np.nanmedian(v))
    return F, cat


def _pct(v: np.ndarray) -> np.ndarray:
    """Percentile 0..1 of each value among non-missing values; missing -> 0.5 (neutral)."""
    out = np.full(len(v), 0.5)
    ok = ~np.isnan(v)
    if ok.sum() > 1:
        r = pd.Series(v[ok]).rank(method="average").to_numpy()
        out[ok] = (r - 1) / (ok.sum() - 1)
    return out


def refine(base_order: np.ndarray, base_score: np.ndarray, comp: pd.DataFrame, F: pd.DataFrame, clar: pd.DataFrame | None,
           filters: dict, boosts: dict, boost_strength: float = 0.5) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply hard filters and soft preferences to an existing ranking.
    base_order: row positions in ranked order; base_score: their scores (higher = better).
    filters: {legal: [..], prefectures: [..], min_clarity: int, founded_from/to: int, exclude_branches: bool,
              factors: {key: {min, max}}}
    boosts: {factor_key: weight -1..1}  (positive = prefer higher values, negative = prefer lower)
    New score = (1 - s) * rank-percentile of the model score + s * weighted factor percentiles, s = boost_strength (only if any boost)."""
    pos = np.asarray(base_order)
    keep = np.ones(len(pos), bool)
    why = {}
    def cut(mask, name):
        nonlocal keep
        before = int(keep.sum()); keep &= mask; why[name] = before - int(keep.sum())
    if filters.get("legal"):
        cut(comp["legal_type"].iloc[pos].isin(filters["legal"]).to_numpy(), "Νομική μορφή")
    if filters.get("prefectures"):
        cut(comp["prefecture_name"].iloc[pos].isin(filters["prefectures"]).to_numpy(), "Νομός")
    if filters.get("min_clarity") and clar is not None:
        cut(clar["clarity"].to_numpy()[pos] >= int(filters["min_clarity"]), "Καθαρότητα ΚΑΔ")
    yr = comp["incorporation_date"].dt.year.to_numpy()[pos]
    if filters.get("founded_from"):
        cut(np.nan_to_num(yr, nan=0) >= int(filters["founded_from"]), "Έτος ίδρυσης")
    if filters.get("founded_to"):
        cut(np.nan_to_num(yr, nan=9999) <= int(filters["founded_to"]), "Έτος ίδρυσης")
    if filters.get("exclude_branches"):
        cut(comp["is_branch"].fillna(0).astype(int).to_numpy()[pos] == 0, "Υποκαταστήματα")
    for k, rng in (filters.get("factors") or {}).items():
        if k not in F:
            continue
        v = F[k].to_numpy()[pos]
        m = np.ones(len(pos), bool)
        if rng.get("min") is not None:
            m &= v >= float(rng["min"])
        if rng.get("max") is not None:
            m &= v <= float(rng["max"])
        cut(m, k)
    pos, sc = pos[keep], np.asarray(base_score, float)[keep]
    active = {k: float(w) for k, w in (boosts or {}).items() if k in F and abs(float(w)) > 1e-9}
    if active and len(pos):
        base_p = _pct(sc)
        tot = sum(abs(w) for w in active.values())
        pref = np.zeros(len(pos))
        for k, w in active.items():
            p = _pct(F[k].to_numpy()[pos].astype(float))
            pref += (abs(w) / tot) * (p if w > 0 else 1 - p)      # w > 0: prefer higher values, w < 0: prefer lower
        new = (1 - boost_strength) * base_p + boost_strength * pref
        o = np.argsort(-new, kind="stable")
        pos, sc = pos[o], new[o]
    return pos, sc, why


def profile_summary(F: pd.DataFrame, cat: list, member_pos: np.ndarray) -> list:
    """For each factor: median among the profile's clients vs the whole market, and a Mann-Whitney test (are clients different?)."""
    from scipy.stats import mannwhitneyu
    out = []
    for c in cat:
        v = F[c["key"]].to_numpy(dtype=float)
        a = v[member_pos]; a = a[~np.isnan(a)]
        b = v[~np.isnan(v)]
        if len(a) < 5:
            continue
        try:
            p = float(mannwhitneyu(a, b).pvalue)
        except ValueError:
            p = 1.0
        out.append({"key": c["key"], "label": c["label"], "unit": c["unit"], "profile_median": round(float(np.median(a)), 3),
                    "market_median": round(float(np.median(b)), 3), "p_value": p, "significant": p < 0.01})
    return out
