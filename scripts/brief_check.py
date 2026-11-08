"""python run.py briefcheck --data data  : runs the analysis and checks the output files against the brief."""
import argparse, sys, tempfile
from backend import analysis as A, briefcheck as BC, config as C, ingest as I, pipeline as P


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run.py briefcheck")
    ap.add_argument("--data", default=str(C.DATA_DIR)); ap.add_argument("--embed", default=C.DEFAULT_EMBED)
    a = ap.parse_args(argv)
    base = P.build_base(a.data, a.embed)
    clients = I.load_clients(I.find_file(a.data, "*Pelatologio*.xlsx"))
    with tempfile.TemporaryDirectory() as d:
        A.run_analysis(P.attach_clients(base, clients), d)
        res = BC.check(d, base.companies, clients["vat"], C.OUT_DIR / "embedding_comparison.json")
    for r in res:
        print(("PASS" if r["passed"] else "FAIL"), r["ref"], r["rule"], "->", r["value"])
    fails = sum(not r["passed"] for r in res)
    print(f"\n{len(res) - fails}/{len(res)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
