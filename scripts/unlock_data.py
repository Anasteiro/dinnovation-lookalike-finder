"""Extract the AES-encrypted data.zip (python's zipfile cannot do AES, pyzipper can).

Usage:  python run.py unlock --zip path/to/data.zip [--password XXXX]
Password can also come from env FORCECRM_ZIP_PASSWORD, otherwise it is asked interactively.
"""
import argparse, getpass, os, sys
from pathlib import Path
import pyzipper

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--password", default=os.environ.get("FORCECRM_ZIP_PASSWORD"))
    a = ap.parse_args(argv)
    pw = a.password or getpass.getpass("data.zip password: ")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    try:
        with pyzipper.AESZipFile(a.zip) as z:
            z.pwd = pw.strip().encode()
            z.extractall(a.out)
            print("Extracted:", *z.namelist(), sep="\n  ")
    except RuntimeError as e:
        sys.exit(f"Could not extract (wrong password?): {e}")

if __name__ == "__main__":
    main()
