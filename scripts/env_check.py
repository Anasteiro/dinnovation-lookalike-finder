"""Beginner-friendly environment check:  python run.py check"""
import json, shutil, subprocess, sys, urllib.request

def line(ok, msg): print(("[OK]   " if ok else "[FAIL] ") + msg)

def main(argv=None):
    line(sys.version_info[:2] in [(3, 11), (3, 12), (3, 13)], f"Python {sys.version.split()[0]} (3.11-3.13 recommended)")
    try:
        import torch
        cuda = torch.cuda.is_available()
        line(True, f"torch {torch.__version__}")
        line(cuda, "GPU visible to torch: " + (torch.cuda.get_device_name(0) if cuda else "NO (embeddings will run on CPU - slower but OK)"))
    except Exception as e:
        line(False, f"torch not importable: {e}")
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        line(True, f"nvidia-smi: {out}")
    else:
        line(False, "nvidia-smi not found (no NVIDIA driver?)")
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        line(True, "Ollama is running. Downloaded models: " + (", ".join(models) or "none yet"))
    except Exception:
        line(False, "Ollama is not reachable at localhost:11434 (start the Ollama app, or run: ollama serve)")
    for mod in ("numpy", "pandas", "scipy", "sklearn", "sentence_transformers", "fastapi", "uvicorn", "h3", "pyzipper"):
        try:
            __import__(mod); line(True, f"import {mod}")
        except Exception as e:
            line(False, f"import {mod} failed: {e}")

if __name__ == "__main__":
    main()
