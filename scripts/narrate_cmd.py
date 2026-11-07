"""Write the AI descriptions into outputs/profiles.json using the chosen local model.
    python run.py narrate --model <ollama-model-name>"""
import argparse, json
from pathlib import Path
from backend import config as C, llm

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--profiles", default=str(C.OUT_DIR / "profiles.json"))
    a = ap.parse_args(argv)
    path = Path(a.profiles)
    profiles = json.loads(path.read_text(encoding="utf-8"))
    gen = llm.ollama_generate(a.model)
    llm.narrate_profiles(profiles, gen, C.CACHE_DIR / "llm")
    path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in profiles:
        m = p["description_meta"]
        print(f"{p['profile_id']}: {m['source']} | verified={m['verified']} | attempts={m['attempts']}\n  {p['description'][:300]}\n")

if __name__ == "__main__":
    main()
