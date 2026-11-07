"""Part 0: data reconnaissance report. Run as soon as the real data is unlocked:
    python run.py profile --data data
Writes outputs/data_report.md and prints it."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from backend import ingest as I, config as C

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_DIR))
    a = ap.parse_args(argv)
    d = Path(a.data)
    u = I.load_universe(d)
    comp, act, desc = u["companies"], u["activity"], u["kad_desc"]
    clients = I.load_clients(I.find_file(d, "*Pelatologio*.xlsx"))
    shops, st = I.match_clients(comp, clients)
    L = []
    p = L.append
    p("# Data report\n")
    p(f"- shops (company rows): **{len(comp):,}**; distinct VATs: **{comp['vat'].nunique():,}**; "
      f"branch rows: {int(comp['is_branch'].sum()):,}")
    p(f"- GPS coverage: {comp['latitude'].notna().mean():.1%}; "
      f"incorporation date range: {comp['incorporation_date'].min()} to {comp['incorporation_date'].max()} "
      f"(unparsed: {comp['incorporation_date'].isna().mean():.1%})")
    p(f"- legal types: " + ", ".join(f"{k} {v:,}" for k, v in comp["legal_type"].value_counts().head(8).items()))
    p(f"- prefectures: {comp['prefecture_name'].nunique()}; municipalities: {comp['municipality_name'].nunique()}")
    per = act.groupby("company_id").size()
    p(f"- activity rows: {len(act):,}; per shop median {per.median():.0f}, p90 {per.quantile(.9):.0f}, max {per.max()}")
    p(f"- activity types: " + ", ".join(f"{k} {v:,}" for k, v in act["atype"].value_counts().items()))
    prim = act[act["atype"] == "primary"]
    p(f"- shops without a primary activity: {len(comp) - prim['company_id'].nunique():,}; distinct primary KADs: {prim['activity_id'].nunique()}")
    sec = act[act["atype"] != "primary"]
    p(f"- non-primary KADs outside division 56 (likely declarative noise): {(~sec['activity_id'].str.startswith('56')).mean():.1%}")
    top = prim["activity_id"].value_counts().head(10)
    p("\n## Top primary KADs\n")
    for k, v in top.items():
        p(f"- {k} · {desc.get(k, '?')[:70]} · {v:,}")
    p("\n## Clients\n")
    p(f"- client VATs in Excel: {st['client_vats']}; matched: {st['matched_vats']}; unmatched: {st['unmatched_vats']}; shops: {st['client_shops']}")
    rev = clients["revenue"].dropna()
    if len(rev):
        p(f"- turnover: median {rev.median():,.0f}, mean {rev.mean():,.0f}, p90 {rev.quantile(.9):,.0f}, max {rev.max():,.0f}; "
          f"top 10% of clients carry {rev.nlargest(max(1, len(rev)//10)).sum()/rev.sum():.0%} of turnover")
    cp = prim[prim["company_id"].isin(shops["company_id"])]["activity_id"].value_counts()
    p(f"- distinct primary KADs among clients: {len(cp)}; top client KAD covers {cp.iloc[0]/cp.sum():.0%} of client shops" if len(cp) else "")
    base = prim["activity_id"].map(cp).fillna(0)
    p(f"- baseline pool: {(base>0).sum():,} shops share a client primary KAD (of {len(comp):,}); "
      f"ties are large, so baseline Recall@k is mostly decided by tie-breaking")
    txt = "\n".join(L)
    C.OUT_DIR.mkdir(exist_ok=True)
    (C.OUT_DIR / "data_report.md").write_text(txt, encoding="utf-8")
    print(txt)

if __name__ == "__main__":
    main()
