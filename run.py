"""One entry point.   python run.py <command> [options]
Commands: serve | getmodel | embedbench | hardsplit | briefcheck | enrich | consistency | check | synth | unlock | profile | cv | tune | analyze | bakeoff | narrate"""
import sys, importlib

for _s in (sys.stdout, sys.stderr):           # Greek text must not crash the Windows console
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CMDS = {"getmodel": "scripts.get_model", "embedbench": "scripts.embed_bench", "briefcheck": "scripts.brief_check", "hardsplit": "scripts.hard_split", "redteam": "scripts.redteam_chat", "evalproduct": "scripts.eval_product", "check": "scripts.env_check", "synth": "scripts.make_synthetic", "unlock": "scripts.unlock_data",
        "profile": "scripts.profile_data", "cv": "scripts.run_cv", "tune": "scripts.tune_cv", "analyze": "scripts.analyze",
        "consistency": "scripts.consistency", "enrich": "backend.enrich", "bakeoff": "scripts.llm_bakeoff", "narrate": "scripts.narrate_cmd", "serve": "scripts.serve"}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        sys.exit(__doc__)
    importlib.import_module(CMDS[sys.argv[1]]).main(sys.argv[2:])
