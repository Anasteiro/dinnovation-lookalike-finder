"""SYNTHETIC data with the exact schema of the challenge files, for development and tests only.
It contains latent customer segments so that the evaluation harness has real signal to find.
NEVER present results from this data as real."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KADS = {  # fake codes + Greek descriptions (synthetic)
    "56101100": "Υπηρεσίες εστιατορίων με πλήρη εξυπηρέτηση",
    "56101200": "Υπηρεσίες ταβερνών και οινομαγειρείων",
    "56101300": "Υπηρεσίες ψαροταβερνών και εστιατορίων θαλασσινών",
    "56102000": "Υπηρεσίες γρήγορης εστίασης και σουβλατζίδικα",
    "56211000": "Υπηρεσίες δείπνου με διανομή και τροφοδοσία εκδηλώσεων",
    "56301100": "Υπηρεσίες καφετεριών και καφέ",
    "56302000": "Υπηρεσίες μπαρ και κέντρων διασκέδασης",
    "56303000": "Υπηρεσίες ζαχαροπλαστείων και αναψυκτηρίων",
    "47111000": "Λιανικό εμπόριο τροφίμων σε μη εξειδικευμένα καταστήματα",
    "55101000": "Υπηρεσίες ξενοδοχείων και παρόμοιων καταλυμάτων",
    "68201200": "Εκμίσθωση και διαχείριση ιδιόκτητων ακινήτων",
    "82301000": "Διοργάνωση συνεδρίων και εκδηλώσεων",
    "96021000": "Υπηρεσίες κομμωτηρίων",
    "47911000": "Λιανικό εμπόριο μέσω διαδικτύου",
}
MUNI = {  # name: (prefecture, lat, lon, kind)
    "ΑΘΗΝΑΙΩΝ": ("ΑΤΤΙΚΗΣ", 37.98, 23.73, "urban"), "ΠΕΙΡΑΙΩΣ": ("ΑΤΤΙΚΗΣ", 37.94, 23.65, "urban"),
    "ΘΕΣΣΑΛΟΝΙΚΗΣ": ("ΘΕΣΣΑΛΟΝΙΚΗΣ", 40.64, 22.94, "urban"), "ΠΑΤΡΕΩΝ": ("ΑΧΑΪΑΣ", 38.25, 21.73, "urban"),
    "ΗΡΑΚΛΕΙΟΥ": ("ΗΡΑΚΛΕΙΟΥ", 35.34, 25.13, "urban"), "ΛΑΡΙΣΑΙΩΝ": ("ΛΑΡΙΣΗΣ", 39.64, 22.42, "urban"),
    "ΜΥΚΟΝΟΥ": ("ΚΥΚΛΑΔΩΝ", 37.45, 25.33, "island"), "ΘΗΡΑΣ": ("ΚΥΚΛΑΔΩΝ", 36.42, 25.43, "island"),
    "ΡΟΔΟΥ": ("ΔΩΔΕΚΑΝΗΣΟΥ", 36.43, 28.22, "island"), "ΧΕΡΣΟΝΗΣΟΥ": ("ΗΡΑΚΛΕΙΟΥ", 35.32, 25.39, "island"),
    "ΚΑΣΣΑΝΔΡΑΣ": ("ΧΑΛΚΙΔΙΚΗΣ", 40.05, 23.42, "island"), "ΒΟΛΟΥ": ("ΜΑΓΝΗΣΙΑΣ", 39.36, 22.94, "urban"),
}
SEGMENTS = {  # kads, municipality kinds, legal types, age center, universe share, client share
    "tavern":  (["56101200", "56101100"], "urban", ["ΑΤΟΜΙΚΗ", "ΟΕ"], 4.5, 0.30, 0.55),
    "seaside": (["56101300", "56302000", "56101100"], "island", ["ΙΚΕ", "ΑΤΟΜΙΚΗ"], 3.0, 0.15, 0.30),
    "cafe":    (["56301100", "56303000"], "urban", ["ΙΚΕ", "ΕΕ"], 2.5, 0.30, 0.10),
    "fast":    (["56102000", "56211000"], "any", ["ΑΤΟΜΙΚΗ"], 4.0, 0.25, 0.05),
}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=24424)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    segs = list(SEGMENTS)
    seg = rng.choice(segs, size=a.n, p=[SEGMENTS[s][4] for s in segs])
    munis = list(MUNI)
    rows, acts = [], []
    for i in range(a.n):
        kads, kind, legals, age_c, _, _ = SEGMENTS[seg[i]]
        pool = [m for m in munis if kind == "any" or MUNI[m][3] == kind]
        if rng.random() < 0.15: pool = munis                      # leakage across kinds
        m = pool[rng.integers(len(pool))]
        pref, la, lo, _ = MUNI[m]
        has_gps = rng.random() < 0.96
        age = float(np.clip(rng.normal(age_c, 1.2), 2.0, 6.9))
        date = pd.Timestamp("2026-11-06") - pd.Timedelta(days=int(age * 365.25))
        lt = legals[rng.integers(len(legals))] if rng.random() < 0.8 else rng.choice(["ΑΤΟΜΙΚΗ", "ΟΕ", "ΕΕ", "ΙΚΕ", "ΑΕ", "ΕΠΕ"])
        cid = f"C{i:06d}"
        rows.append({"company_id": cid, "vat": f"{100000000 + i // 1:09d}", "company_name_el": f"Μαγαζί {i}",
                     "legal_type": lt, "is_branch": 0, "street": "Οδός", "street_number": "1", "zip_code": "10000",
                     "city": m, "municipality_name": m, "prefecture_name": pref,
                     "incorporation_date": date.strftime("%Y-%m-%d"),
                     "latitude": f"{la + rng.normal(0, 0.04):.6f}".replace(".", ",") if has_gps else "NULL",
                     "longitude": f"{lo + rng.normal(0, 0.04):.6f}".replace(".", ",") if has_gps else "NULL", "_seg": seg[i]})
        main_k = kads[rng.integers(len(kads))] if rng.random() < 0.85 else rng.choice(list(KADS)[:8])
        acts.append({"company_id": cid, "activity_id": main_k, "activity_type": "Κύρια", "date_from": date.strftime("%Y-%m-%d")})
        for k in rng.choice(list(KADS), size=int(rng.choice([0, 0, 1, 2, 5, 9])), replace=False):  # declarative noise
            if k != main_k:
                acts.append({"company_id": cid, "activity_id": k, "activity_type": rng.choice(["Δευτερεύουσα", "Βοηθητική"]),
                             "date_from": date.strftime("%Y-%m-%d")})
    comp = pd.DataFrame(rows)
    # branches: ~6% of shops are extra shops of an existing VAT
    br = rng.choice(a.n, size=int(a.n * 0.06), replace=False)
    comp.loc[br[: len(br) // 2], "vat"] = comp.loc[br[len(br) // 2:], "vat"].values[: len(br) // 2]
    comp.loc[br[: len(br) // 2], "is_branch"] = 1
    segcol = comp.pop("_seg")
    comp.to_csv(out / "vwHackCompany.csv", sep=";", index=False, encoding="utf-8")
    pd.DataFrame(acts).to_csv(out / "vwHackCompanyActivity.csv", sep=";", index=False, encoding="utf-8")
    pd.DataFrame({"activity_id": list(KADS), "description": list(KADS.values())}).to_csv(
        out / "vwHackActivities.csv", sep=";", index=False, encoding="utf-8")
    # clients: 300 VATs drawn with segment-specific propensity
    cs = np.array([SEGMENTS[s][5] / SEGMENTS[s][4] for s in segcol])
    p = cs / cs.sum()
    pick = rng.choice(a.n, size=300, replace=False, p=p)
    vats = comp.loc[pick, "vat"].drop_duplicates().tolist()
    rev = np.round(np.exp(rng.normal(9.6, 0.9, len(vats))), -1)
    pd.DataFrame({"ΑΦΜ": vats, "Τζίρος (€)": rev}).to_excel(out / "ForceCRM_Hackathon_Pelatologio.xlsx", index=False)
    (out / "SYNTHETIC.txt").write_text("Synthetic development data. Not real.\n")
    print(f"synthetic data written to {out} ({a.n} shops, {len(vats)} client VATs)")

if __name__ == "__main__":
    main()
