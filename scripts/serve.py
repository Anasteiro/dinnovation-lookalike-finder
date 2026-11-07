"""Start the whole app on localhost (the one command):   python run.py serve
Options: --data data --embed tfidf --llm-model <ollama name> --port 8000 --no-browser"""
import argparse, threading, webbrowser
import uvicorn
from backend import config as C
from backend.api import create_app

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_DIR))
    ap.add_argument("--embed", default="tfidf")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    app = create_app(data_dir=a.data, embed=a.embed, llm_model=a.llm_model)
    threading.Thread(target=lambda: _warm(app), daemon=True).start()      # build market data while the page loads
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{a.port}")).start()
    print(f"App running at http://localhost:{a.port}  (Ctrl+C to stop). Everything runs locally.")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")

def _warm(app):
    try:
        app.state.warmup()
        print("market data ready")
    except Exception as e:
        print("warm-up failed (data not unlocked yet?):", e)

if __name__ == "__main__":
    main()
