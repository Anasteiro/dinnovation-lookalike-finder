"""End-to-end accuracy of the PRODUCTION pipeline (exactly what the app/CLI produce), on the real client list.
    python run.py evalproduct --data data --repeats 10
Each repeat hides 20% of the client VATs, runs analysis.run_analysis() on the rest, and checks where the hidden clients
appear in the deliverable ranking. Reports Recall@500/2000 (shop-level and VAT-level) for:
  product   = lookalikes_global order (= the order written to lookalikes.csv)
  baseline  = same primary KAD as the training clients, by KAD frequency (the brief's baseline)
plus how many rows a reader would take with 'rank <= k' in lookalikes.csv (must be exactly k)."""
import argparse, time
import numpy as np, pandas as pd
from backend import analysis as A, evalharness as H, pipeline as P, ingest as I, config as C


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run.py evalproduct")
    ap.add_argument("--data", default=str(C.DATA_DIR)); ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--embed", default="tfidf"); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    t = time.time(); base = P.build_base(a.data, a.embed); print(f"market built in {time.time() - t:.1f}s")
    clients = I.load_clients(I.find_file(a.data, "*Pelatologio*.xlsx"))
    vats = clients["vat"].unique(); rng = np.random.default_rng(a.seed)
    comp = base.companies; pos_of = pd.Series(np.arange(len(comp)), index=comp["company_id"].to_numpy())
    vat_arr = comp["vat"].to_numpy(); rows = []
    for r in range(a.repeats):
        hid = set(rng.permutation(vats)[: int(round(0.2 * len(vats)))])
        ctx = P.attach_clients(base, clients[~clients["vat"].isin(hid)])
        t0 = time.time(); out = A.run_analysis(ctx); secs = time.time() - t0
        hid_pos = np.flatnonzero(np.isin(vat_arr, list(hid)))
        g = out["lookalikes_global"]; gpos = pos_of[g["company_id"].to_numpy()].to_numpy()
        rank = np.full(len(comp), 10 ** 9); rank[gpos] = np.arange(len(gpos))
        base_sc = H.baseline_scorer(ctx.primary_kad)(ctx.client_shops["pos"].to_numpy(), None)
        border = H.rank_order(base_sc, ctx.client_shops["pos"].to_numpy(), np.random.default_rng(r))
        brank = np.empty(len(comp), np.int64); brank[border] = np.arange(len(comp))
        rk_csv = g["rank"].to_numpy()                     # the `rank` column exactly as written to lookalikes.csv
        row = {"repeat": r, "k_profiles": out["meta"].get("k", len(out["profiles"])), "seconds": round(secs, 2)}
        for k in C.KS:
            row[f"product@{k}"] = float((rank[hid_pos] < k).mean())
            row[f"baseline@{k}"] = float((brank[hid_pos] < k).mean())
            row[f"product_vat@{k}"] = float(pd.Series(rank[hid_pos] < k).groupby(vat_arr[hid_pos]).any().mean())
            row[f"rows_with_rank<={k}"] = int((rk_csv <= k).sum())
        rows.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    df = pd.DataFrame(rows)
    print("\nmean over repeats (95% CI):")
    for c in df.columns[1:]:
        m, s = df[c].mean(), df[c].std(ddof=1) / np.sqrt(len(df)) * 1.96
        print(f"  {c:22s} {m:8.3f}  ±{s:.3f}")
    return df


if __name__ == "__main__":
    main()
