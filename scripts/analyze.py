"""Run the full analysis and write the files the organisers grade.
    python run.py analyze --data data --out outputs --embed tfidf"""
import argparse
from pathlib import Path
from backend import config as C
from backend.pipeline import build_context
from backend.analysis import run_analysis

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_DIR))
    ap.add_argument("--out", default=str(C.OUT_DIR))
    ap.add_argument("--embed", default=C.DEFAULT_EMBED)
    ap.add_argument("--k", type=int, default=None, help="force the number of profiles (default: automatic)")
    ap.add_argument("--boot", type=int, default=30)
    ap.add_argument("--match", default="split", choices=["split", "head", "full"])
    a = ap.parse_args(argv)
    ctx = build_context(Path(a.data), embed=a.embed, match_mode=a.match, fallback=True)
    res = run_analysis(ctx, Path(a.out), k=a.k, B=a.boot, progress=lambda s, **kw: print(" ..", s))
    m = res["meta"]
    print(f"\nprofiles: {m['k']} ({m['selection_rule']})  | none-share {m['none_share']:.1%} | {m['timings']}")
    for p in res["profiles"]:
        print(f"- {p['profile_id']}: {p['title']} | clients {p['n_clients']} | stability {p['stability']} | features {len(p['features'])}")
        print("   ", p["description"][:230])
    print(f"files written to {a.out}")

if __name__ == "__main__":
    main()
