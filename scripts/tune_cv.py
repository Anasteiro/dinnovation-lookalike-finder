"""Tuning sweeps on the real data (stage 1: ranking recipe, stage 2: block weights, stage 3: embeddings).
    python run.py tune --data data --embed tfidf --stage 1 --repeats 12
Selection uses mean Recall@2000 and @500 with paired comparison to the baseline; do not over-tune on 300 clients."""
import argparse, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from backend import config as C, evalharness as H, rankers as R
from backend.pipeline import build_context

W0 = dict(C.DEFAULT_BLOCK_WEIGHTS)

def variants(stage):
    S = {}
    if stage == 1:
        for m in (3, 5, 10, 20, 40):
            S[f"knn m={m}"] = ("knn", m, 0.0, None)
        for m in (5, 10, 20):
            S[f"PRODUCT k=3 blend=0 m={m} ecdf"] = ("prod", 3, 0.0, m, "ecdf", None)
            S[f"PRODUCT k=3 blend=0 m={m} z"] = ("prod", 3, 0.0, m, "z", None)
        S["PRODUCT k=1 blend=0 m=10 raw"] = ("prod", 1, 0.0, 10, "raw", None)
        S["PRODUCT k=5 blend=0 m=10 ecdf"] = ("prod", 5, 0.0, 10, "ecdf", None)
    elif stage == 2:
        def w(**kw):
            d = dict(W0); d.update(kw); return d
        for name, d in {"default": W0, "geo 1.0": w(geo=1.0), "geo 0.3": w(geo=0.3), "admin 0.8": w(admin=0.8), "admin 0": w(admin=0.0),
                        "legal 0.6": w(legal=0.6), "legal 0.15": w(legal=0.15), "age 0.6": w(age=0.6), "age 0": w(age=0.0),
                        "struct 0.4": w(struct=0.4), "struct 0": w(struct=0.0), "activity 1.5": w(activity=1.5), "activity 0.6": w(activity=0.6)}.items():
            S[f"knn m=5 | {name}"] = ("knn", 5, 0.0, d)
    elif stage == 3:
        def w(**kw):
            d = dict(W0); d.update(kw); return d
        combos = {"default": W0, "A admin0 geo.3 legal.6": w(admin=0, geo=.3, legal=.6), "B admin0 geo.6 legal.6": w(admin=0, geo=.6, legal=.6),
                  "C admin.1 geo.3 legal.6": w(admin=.1, geo=.3, legal=.6), "D admin0 geo.3 legal.9": w(admin=0, geo=.3, legal=.9),
                  "E admin0 geo0 legal.6": w(admin=0, geo=0, legal=.6), "F admin0 geo.3 legal.6 age.15": w(admin=0, geo=.3, legal=.6, age=.15),
                  "G admin0 geo.15 legal.6": w(admin=0, geo=.15, legal=.6), "H admin0 geo.3 legal.6 struct.2": w(admin=0, geo=.3, legal=.6, struct=.2)}
        for name, d in combos.items():
            S[f"knn m=5 | {name}"] = ("knn", 5, 0.0, d)
            S[f"PROD k=3 m=5 ecdf | {name}"] = ("prodw", 3, 0.0, 5, "ecdf", d)
    elif stage == 4:
        I = dict(W0); I.update(admin=.1, geo=.3, legal=.6, age=.15)
        for m in (3, 5, 8, 12):
            S[f"global knn m={m}"] = ("knn", m, 0.0, I)
        for k in (3, 5):
            for how in ("raw", "z", "ecdf"):
                S[f"PROD k={k} m=5 {how}"] = ("prodw", k, 0.0, 5, how, I)
    return S

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_DIR)); ap.add_argument("--embed", default="tfidf")
    ap.add_argument("--stage", type=int, default=1); ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--match", default="split"); ap.add_argument("--revenue", default="log")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    t0 = time.time()
    ctx = build_context(Path(a.data), embed=a.embed, match_mode=a.match)
    s = ctx.client_shops
    w = R.revenue_weights(s["revenue"].to_numpy(), a.revenue)
    sc = {"baseline": H.baseline_scorer(ctx.primary_kad)}
    for name, v in variants(a.stage).items():
        if v[0] == "knn":
            sc[name] = R.knn_scorer(ctx.model, v[1], v[2], v[3])
        elif v[0] == "prodw":
            sc[name] = R.profile_scorer(ctx.model, k=v[1], blend=v[2], m_knn=v[3], how=v[4], weights=v[5])
        else:
            sc[name] = R.profile_scorer(ctx.model, k=v[1], blend=v[2], m_knn=v[3], how=v[4])
    df = H.cross_validate(len(ctx.companies), s["pos"].to_numpy(), w, s["vat"].to_numpy(), sc, repeats=a.repeats)
    out = H.summarize(df)
    pd.set_option("display.width", 200)
    print(out[["model", "recall@500", "recall@500_ci95", "recall@2000", "recall@2000_ci95", "lift_vs_base@2000", "paired_wins@2000"]].round(4).to_string(index=False))
    C.OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(C.OUT_DIR / f"tune_stage{a.stage}_{a.embed.replace(':','_').replace('/','_')}.csv", index=False)
    print(f"({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
