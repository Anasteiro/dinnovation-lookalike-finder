"""Full hard-split report on the real client list.
    python run.py hardsplit --data data [--quick]
Writes outputs/hardsplit/report.json and report.md."""
import argparse, json
from pathlib import Path
from backend import config as C, hardsplit as HS, ingest as I, pipeline as P


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def to_md(r: dict) -> str:
    L = r["locked"]; rp = r["repeated"]
    t = ["# Hard-split report", "", f"Split {r['split']['fingerprint']} (salt `{r['split']['salt']}`), sizes {r['split']['sizes']}. Runtime {r['seconds']} s.", "",
         "## Locked test (20% of VATs)", "", "| | Recall@500 | Recall@2000 |", "|---|---|---|",
         f"| Product | {pct(L['product@500'])} [{pct(L['product@500_ci95'][0])}, {pct(L['product@500_ci95'][1])}] | {pct(L['product@2000'])} [{pct(L['product@2000_ci95'][0])}, {pct(L['product@2000_ci95'][1])}] |",
         f"| Baseline | {pct(L['baseline@500'])} | {pct(L['baseline@2000'])} |", f"| Chance | {pct(L['chance@500'])} | {pct(L['chance@2000'])} |", "",
         f"## Other hash salts ({rp['repeats']} repeats)", "", f"Recall@500 {pct(rp['product@500'])} (sd {pct(rp.get('product@500_sd'))}), Recall@2000 {pct(rp['product@2000'])} (sd {pct(rp.get('product@2000_sd'))}); wins over baseline {rp['wins@500']}/{rp['repeats']} and {rp['wins@2000']}/{rp['repeats']}.", ""]
    if r["generalisation"]:
        t += ["## Deliberately hard splits", "", "| Split | train | test | R@500 | R@2000 | base R@500 | base R@2000 | chance R@2000 |", "|---|---|---|---|---|---|---|---|"]
        for g in r["generalisation"]:
            t.append(f"| {g['name']} | skipped: {g['skipped']} | | | | | | |" if "skipped" in g else f"| {g['name']} | {g['n_train']} | {g['n_test']} | {pct(g['product@500'])} | {pct(g['product@2000'])} | {pct(g['baseline@500'])} | {pct(g['baseline@2000'])} | {pct(g['chance@2000'])} |")
        t.append("")
    t += ["## Evaluation canaries", ""] + [f"- {'PASS' if c['passed'] else 'FAIL'}: {c['name']}" + (f" (observed {c['observed']}, expected {c['expected']})" if "observed" in c else "") for c in r["canaries"]] + [""]
    fb = r["feedback_blind"][0]
    t += ["## Feedback, blind oracle", "", f"{fb['reviewed']} reviewed, {fb['accepted']} accepted, {fb['test_clients_left']} test clients left in the list.", "",
          "| | Recall@500 | Recall@2000 |", "|---|---|---|", f"| Without feedback | {pct(fb['without@500'])} | {pct(fb['without@2000'])} |", f"| With feedback | {pct(fb['with@500'])} | {pct(fb['with@2000'])} |", "",
          "## Adapting to a reviewer's intent (simulated reviewer)", "", "Share of the next 100 proposals that match the intent.", "", "| Intent | Model only | Feedback, generic rejection | Feedback, with reasons |", "|---|---|---|---|"]
    t += [f"| {x['intent']} | {pct(x['model_p100'])} | {pct(x.get('generic_p100'))} | {pct(x['reasons_p100'])} |" for x in r["intent"]]
    t = ["## Συμπεράσματα", ""] + [f"- {x['text']}" for x in r["insights"]] + [""] + t
    return "\n".join(t + ["", r["note"], ""])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run.py hardsplit")
    ap.add_argument("--data", default=str(C.DATA_DIR)); ap.add_argument("--embed", default=C.DEFAULT_EMBED); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(C.OUT_DIR / "hardsplit"))
    a = ap.parse_args(argv)
    base = P.build_base(a.data, a.embed)
    clients = I.load_clients(I.find_file(a.data, "*Pelatologio*.xlsx"))
    r = HS.run_suite(base, clients, quick=a.quick, progress=lambda e: print(f"[{e['pct']:3d}%] {e['message']}", flush=True))
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "report.md").write_text(to_md(r), encoding="utf-8")
    print(to_md(r))


if __name__ == "__main__":
    main()
