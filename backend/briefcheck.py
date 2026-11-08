"""Automatic check of the output files against the technical brief (sections 8.1, 10 and A2/A3/A7).
Reports each rule as passed or failed with the measured value, so a wrong file is caught before the organisers see it."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

LOOK = ["profile_id", "company_id", "rank", "score", "αιτιολόγηση"]
ASSIGN = ["company_id", "profile_id", "score"]
PROFILE_KEYS = ["profile_id", "title", "description", "n_clients", "turnover_share", "stability", "confidence", "features"]
FORBIDDEN = ("πιστοληπτ", "φερέγγυ", "αξιόπιστ")


def check(out_dir, companies: pd.DataFrame, client_vats, comparison: Path | None = None) -> list[dict]:
    d = Path(out_dir)
    res = []

    def rule(ref, text, ok, value=""):
        res.append({"ref": ref, "rule": text, "passed": bool(ok), "value": str(value)})

    ids = companies["company_id"].astype(str)
    clients = set(ids[companies["vat"].isin(set(client_vats))])
    look = pd.read_csv(d / "lookalikes.csv", dtype=str, encoding="utf-8")
    assign = pd.read_csv(d / "assignments.csv", dtype=str, encoding="utf-8", keep_default_na=False)
    prof = json.loads((d / "profiles.json").read_text(encoding="utf-8"))
    rank, score = pd.to_numeric(look["rank"]), pd.to_numeric(look["score"])
    pids = {p["profile_id"] for p in prof}

    rule("§10", "lookalikes.csv: στήλες profile_id, company_id, rank, score, αιτιολόγηση", list(look.columns) == LOOK, list(look.columns))
    rule("§8.1", "lookalikes.csv: τουλάχιστον 2.000 προτάσεις", len(look) >= 2000, len(look))
    rule("§8.1", "Μοναδικό rank 1..N στη σειρά του αρχείου (οι πρώτες k θέσεις είναι σαφείς)", list(rank) == list(range(1, len(look) + 1)), f"{rank.min()}..{rank.max()}")
    rule("§8.1", "Το score δεν ανεβαίνει όσο πέφτει η θέση", (score.diff().dropna() <= 0).all(), f"{score.max()}..{score.min()}")
    rule("§4 Α7", "Κάθε επιχείρηση μία φορά", look["company_id"].is_unique, look["company_id"].nunique())
    rule("§4 Α7", "Καμία πρόταση δεν είναι ήδη πελάτης", not (set(look["company_id"]) & clients), len(set(look["company_id"]) & clients))
    rule("§4 Α7", "Όλες οι προτάσεις υπάρχουν στο universe", set(look["company_id"]) <= set(ids))
    rule("§4 Α7", "Κάθε πρόταση έχει αιτιολόγηση", look["αιτιολόγηση"].fillna("").str.len().gt(0).all())
    rule("§4 Α7", "profile_id κάθε πρότασης υπάρχει στο profiles.json", set(look["profile_id"]) <= pids, sorted(set(look["profile_id"])))
    rule("§10", "assignments.csv: στήλες company_id, profile_id, score", list(assign.columns) == ASSIGN, list(assign.columns))
    rule("§4 Α3", "Μία ανάθεση για κάθε επιχείρηση του universe", len(assign) == len(ids) and assign["company_id"].is_unique, len(assign))
    rule("§4 Α3", "Υπάρχει η έκβαση «κανένα profile» και κάθε άλλη τιμή είναι profile", set(assign["profile_id"]) <= pids | {"none"}, sorted(set(assign["profile_id"])))
    rule("§4 Α2", "1 έως 6 profiles", 1 <= len(prof) <= 6, len(prof))
    missing = [k for p in prof for k in PROFILE_KEYS if k not in p]
    rule("§10", "profiles.json: τίτλος, περιγραφή, πλήθος, μερίδιο τζίρου, σταθερότητα, εμπιστοσύνη, χαρακτηριστικά", not missing, missing)
    feats = [f for p in prof for f in p.get("features", [])]
    rule("§4 Α4", "Κάθε χαρακτηριστικό έχει lift και p-value", feats and all("lift" in f and "p_value" in f for f in feats), len(feats))
    rule("§4 Α6", "Κάθε περιγραφή σημαίνεται ως AI ή όχι", all("description_is_ai" in p for p in prof), [p.get("description_is_ai") for p in prof])
    text = (d / "profiles.json").read_text(encoding="utf-8").lower()
    bad = [w for w in FORBIDDEN if w in text] + [w for w in FORBIDDEN if look["αιτιολόγηση"].fillna("").str.lower().str.contains(w, regex=False).any()]
    rule("§9", "Καμία διατύπωση φερεγγυότητας", not bad, bad)
    if comparison is not None:
        r = json.loads(comparison.read_text(encoding="utf-8")) if comparison.exists() else {}
        n = int(r.get("neural_done", 0)); total = sum(m.get("status") == "done" for m in r.get("models", []))
        rule("§5.3", "Μετρημένη σύγκριση embeddings: τουλάχιστον δύο επιλογές, με νευρωνικό μοντέλο", n >= 1 and total >= 2, f"{total} μοντέλα, {n} νευρωνικά")
    return res
