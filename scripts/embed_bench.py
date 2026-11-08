"""Mandatory comparison of embedding models on the real data (brief section 5.3).
    python run.py embedbench --data data          or double-click compare_embeddings.bat, or the page "Σύγκριση embeddings" in the app
Writes outputs/embedding_comparison.md and .json. Models download once, then run locally. Exit code 1 if no neural model could run."""
import argparse, sys
from pathlib import Path
from backend import config as C, embedbench as EB, ingest as I


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run.py embedbench")
    ap.add_argument("--data", default=str(C.DATA_DIR)); ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=EB.DEFAULT_BACKENDS); ap.add_argument("--out", default=str(C.OUT_DIR))
    a = ap.parse_args(argv)
    pf = EB.preflight()
    if not pf["ready"] and any(m.startswith("st:") for m in a.models):
        print("\nΔΕΝ ΜΠΟΡΕΙ ΝΑ ΞΕΚΙΝΗΣΕΙ Η ΣΥΓΚΡΙΣΗ\n" + pf["fix"]); sys.exit(1)
    clients = I.load_clients(I.find_file(a.data, "*Pelatologio*.xlsx"))
    seen = [0]

    def show(e):
        for line in e["log"][-max(1, e["done"] - seen[0]):] if e["done"] > seen[0] else []:
            print(f"[{e['pct']:3d}%] {line}", flush=True)
        seen[0] = e["done"]
    r = EB.run(a.data, clients, a.out, a.models, a.repeats, show)
    print("\n" + (Path(a.out) / "embedding_comparison.md").read_text(encoding="utf-8"))
    for x in r["insights"]:
        print("-", x["text"])
    if any(m.startswith("st:") for m in a.models) and not r["neural_done"]:
        print("\nΚΑΝΕΝΑ ΝΕΥΡΩΝΙΚΟ ΜΟΝΤΕΛΟ ΔΕΝ ΕΚΤΕΛΕΣΤΗΚΕ. Δείτε τους λόγους παραπάνω."); sys.exit(1)


if __name__ == "__main__":
    main()
