"""Compare local LLMs on the SAME profiles: speed, memory-free metrics, groundedness, and the Greek text itself
(so a native speaker can judge quality). Output: outputs/llm_bakeoff.md  (+ .json).
    python run.py bakeoff --models hf.co/QuantFactory/EuroLLM-9B-Instruct-GGUF:Q4_K_M MHKetbi/ilsp-Llama-Krikri-8B-Instruct:q4_K_S
Needs outputs/profiles.json (run `python run.py analyze ...` first)."""
import argparse, json, urllib.request
from pathlib import Path
from backend import config as C, llm

def available(host):
    with urllib.request.urlopen(host + "/api/tags", timeout=5) as r:
        return {m["name"] for m in json.loads(r.read()).get("models", [])}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--profiles", default=str(C.OUT_DIR / "profiles.json"))
    ap.add_argument("--n", type=int, default=3, help="how many profiles to narrate per model")
    ap.add_argument("--host", default=llm.OLLAMA_URL)
    a = ap.parse_args(argv)
    profiles = json.loads(Path(a.profiles).read_text(encoding="utf-8"))[: a.n]
    try:
        have = available(a.host)
    except Exception as e:
        raise SystemExit(f"Ollama is not reachable ({e}). Start it first.")
    rows, md = [], ["# LLM bake-off\n", "Fixed options: " + json.dumps(llm.GEN_OPTIONS), ""]
    for m in a.models:
        if not any(h == m or h.startswith(m) or m.startswith(h.split(":")[0]) and m.split(":")[-1] in h for h in have):
            print(f"SKIP {m}: not downloaded (ollama pull {m})"); continue
        gen = llm.ollama_generate(m, a.host)
        try:
            gen("Πες «έτοιμος».", "Απάντησε σύντομα.")          # warm-up, loads the model into memory
        except Exception as e:
            print(f"SKIP {m}: {e}"); continue
        res = [llm.narrate(p, gen, None) for p in profiles]
        ok = sum(r["verified"] for r in res)
        par = [q["is_ai"] for r in res for q in r["paragraphs"]]
        tps = [r["stats"].get("tokens_per_s") for r in res if r["stats"].get("tokens_per_s")]
        sec = [r["stats"].get("seconds", 0) for r in res]
        row = {"model": m, "profiles_fully_verified": f"{ok}/{len(res)}", "paragraphs_verified": f"{sum(par)}/{len(par)}",
               "paragraphs_ok_first_try": f"{sum(q['attempts'] == 1 and q['is_ai'] for r in res for q in r['paragraphs'])}/{len(par)}", "avg_seconds_per_profile": round(sum(sec) / len(sec), 1),
               "avg_tokens_per_s": round(sum(tps) / len(tps), 1) if tps else None, "avg_attempts": round(sum(r["attempts"] for r in res) / len(res), 2)}
        rows.append(row); print(row)
        md += [f"## {m}", json.dumps(row, ensure_ascii=False), ""]
        for r in res:
            md += [f"### {r['profile_id']}  (source={r['source']}, verified={r['verified']})"]
            for q in r["paragraphs"]:
                md += [f"**{q['title']}** - {'AI' if q['is_ai'] else 'FALLBACK (no AI)'}, attempts={q['attempts']}", q["text"],
                       "" if q["is_ai"] else "Violations: " + "; ".join(q["violations"][:3]), ""]
            md += [
                   "Greek quality (1-5): ___   Fluency: ___   Sounds natural to a Greek salesperson: ___", ""]
    C.OUT_DIR.mkdir(exist_ok=True)
    (C.OUT_DIR / "llm_bakeoff.md").write_text("\n".join(md), encoding="utf-8")
    (C.OUT_DIR / "llm_bakeoff.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten: outputs/llm_bakeoff.md  <- open it and rate the Greek text")

if __name__ == "__main__":
    main()
