"""Downloads the embedding model once and stores the KAD embeddings, so the app needs neither the model nor the internet afterwards.
    python run.py getmodel [--embed st:<model>]"""
import argparse, time
from backend import config as C, embed as E, ingest as I


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run.py getmodel")
    ap.add_argument("--data", default=str(C.DATA_DIR)); ap.add_argument("--embed", default=C.DEFAULT_EMBED)
    a = ap.parse_args(argv)
    u = I.load_universe(a.data)
    kad_ids = sorted(set(u["activity"]["activity_id"].dropna()))
    t = time.time()
    v = E.embed_texts([u["kad_desc"].get(k) or k for k in kad_ids], a.embed)
    print(f"OK: {a.embed}, {v.shape[0]} περιγραφές ΚΑΔ, {v.shape[1]} διαστάσεις, {time.time() - t:.0f} δευτ.")


if __name__ == "__main__":
    main()
