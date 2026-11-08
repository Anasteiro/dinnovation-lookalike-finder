"""Reproducibility check: the same client file must always give exactly the same profiles, assignments and lookalikes.
    python run.py consistency --data data
Runs the full analysis several times (each from scratch, one with the client rows shuffled) and compares file fingerprints."""
import argparse, hashlib, shutil, tempfile
from pathlib import Path
import pandas as pd
from backend import config as C, analysis as A, ingest as I
from backend.pipeline import build_context

FILES = ("assignments.csv", "assignments_detail.csv", "lookalikes.csv", "lookalikes_by_profile.csv", "lookalikes_global.csv", "profiles.json")


def fingerprint(d: Path):
    return {f: hashlib.md5((d / f).read_bytes()).hexdigest()[:10] for f in FILES}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_DIR))
    ap.add_argument("--embed", default=C.DEFAULT_EMBED)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args(argv)
    data = Path(a.data); tmp = Path(tempfile.mkdtemp())
    prints = []
    xlsx = I.find_file(data, "*Pelatologio*.xlsx")
    for i in range(a.runs):
        src = xlsx
        if i == a.runs - 1:                                   # last run: same clients, rows in a different order
            df = pd.read_excel(xlsx).sample(frac=1, random_state=i).reset_index(drop=True)
            src = tmp / "shuffled.xlsx"; df.to_excel(src, index=False)
        ctx = build_context(data, src, embed=a.embed)
        A.run_analysis(ctx, tmp / f"run{i}")
        prints.append(fingerprint(tmp / f"run{i}"))
        print(f"run {i + 1}/{a.runs}{' (rows shuffled)' if i == a.runs - 1 else ''}:", prints[-1])
    ok = all(p == prints[0] for p in prints)
    print("\nRESULT:", "IDENTICAL in all runs - fully reproducible" if ok else "DIFFERENCES FOUND")
    shutil.rmtree(tmp, ignore_errors=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
