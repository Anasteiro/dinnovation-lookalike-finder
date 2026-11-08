"""Part 2/3: cross-validated Recall@k of the baseline vs similarity rankers + ablations.
    python run.py cv --data data --embed tfidf --repeats 30
    python run.py cv --data data --embed st:intfloat/multilingual-e5-base"""
import argparse, time
from pathlib import Path
import pandas as pd
from backend import config as C, evalharness as H, rankers as R
from backend.pipeline import build_context

ABLATIONS = {
    "activity only": {"activity": 1},
    "+admin/geo": {"activity": 1, "admin": 0.4, "geo": 0.6},
    "+legal": {"activity": 1, "admin": 0.4, "geo": 0.6, "legal": 0.3},
    "+age (all)": None,   # default weights
}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_DIR))
    ap.add_argument("--embed", default=C.DEFAULT_EMBED)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--match", default="split", choices=["split", "head", "full"])
    ap.add_argument("--revenue", default="log", choices=["log", "sqrt", "none"])
    a = ap.parse_args(argv)
    t0 = time.time()
    ctx = build_context(Path(a.data), embed=a.embed, match_mode=a.match)
    print(f"context built in {time.time()-t0:.1f}s | {ctx.match_stats} | universe {len(ctx.companies):,}")
    s = ctx.client_shops
    w = R.revenue_weights(s["revenue"].to_numpy(), a.revenue)
    scorers = {"baseline": H.baseline_scorer(ctx.primary_kad)}
    for name, wt in ABLATIONS.items():
        scorers[f"centroid | {name}"] = R.centroid_scorer(ctx.model, wt)
    for name, wt in ABLATIONS.items():
        scorers[f"knn5 | {name}"] = R.knn_scorer(ctx.model, 5, 0.0, wt)
    for k in (1, 3, 5):
        for how in (("raw",) if k == 1 else ("raw", "z", "ecdf")):
            scorers[f"PRODUCT k={k} merge={how}"] = R.profile_scorer(ctx.model, k=k, how=how)
    df = H.cross_validate(len(ctx.companies), s["pos"].to_numpy(), w, s["vat"].to_numpy(), scorers, repeats=a.repeats)
    summ = H.summarize(df)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(summ.round(4).to_string(index=False))
    C.OUT_DIR.mkdir(exist_ok=True)
    summ.to_csv(C.OUT_DIR / "cv_results.csv", index=False)
    print(f"saved outputs/cv_results.csv ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()
