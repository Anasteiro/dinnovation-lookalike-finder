"""Embedding comparison for brief section 5.3: the same pipeline and the same hold-out splits with each embedding backend.
Measured on the challenge data: Greek quality of the KAD neighbours, end-to-end Recall@k, memory and time.
Licence and origin are listed for each model and must be confirmed on the specific repository (the brief asks for that)."""
from __future__ import annotations
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C, embed as E, hardsplit as HS, ingest as I, pipeline as P, rankers as R

DEFAULT_BACKENDS = ["tfidf", "st:intfloat/multilingual-e5-base", "st:BAAI/bge-m3"]
INFO = {
    "tfidf": ("TF-IDF χαρακτήρων + SVD (χωρίς νευρωνικό δίκτυο)", "scikit-learn, BSD-3", "Ανοιχτό λογισμικό, χωρίς εκπαιδευμένο μοντέλο"),
    "st:intfloat/multilingual-e5-base": ("multilingual-e5-base", "MIT", "ΗΠΑ (Microsoft)"),
    "st:intfloat/multilingual-e5-large": ("multilingual-e5-large", "MIT", "ΗΠΑ (Microsoft)"),
    "st:BAAI/bge-m3": ("BGE-M3", "MIT", "Κίνα (BAAI)"),
    "st:ibm-granite/granite-embedding-278m-multilingual": ("granite-embedding multilingual", "Apache-2.0", "ΗΠΑ (IBM)"),
}


def peak_memory_mb() -> float | None:
    """Peak memory of this process so far; None if the platform gives no way to read it."""
    try:
        import resource
        import sys
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(v / (1024 ** 2 if sys.platform == "darwin" else 1024), 0)
    except ImportError:
        pass
    try:
        import psutil
        info = psutil.Process(os.getpid()).memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 1024 ** 2, 0)
    except ImportError:
        return None


def kad_neighbour_quality(emb: np.ndarray, kad_ids: list[str], digits: int = 4, m: int = 5) -> float:
    """Greek quality on the challenge's own texts: of the m nearest descriptions to each KAD description, the share that belong to
    the same official KAD class (first `digits` digits). The classification is independent of the text similarity being tested."""
    groups = pd.Series(kad_ids).astype(str).str[:digits]
    counts = groups.map(groups.value_counts()).to_numpy()
    idx = np.flatnonzero(counts > 1)                     # a class with a single code has no possible correct neighbour
    if len(idx) == 0:
        return float("nan")
    X = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    g = groups.to_numpy()
    hits, m = 0, min(m, len(emb) - 1)
    for start in range(0, len(idx), 512):
        b = idx[start:start + 512]
        S = X[b] @ X.T
        S[np.arange(len(b)), b] = -np.inf
        top = np.argpartition(-S, m - 1, axis=1)[:, :m]
        hits += int((g[top] == g[b][:, None]).sum())
    return hits / (len(idx) * m)


@dataclass
class Row:
    backend: str
    status: str
    detail: dict


def preflight() -> dict:
    """Can neural models run at all in THIS Python? The usual failure is a different Python or a missing package."""
    import importlib.util
    import sys
    missing = [m for m in ("sentence_transformers", "torch") if importlib.util.find_spec(m) is None]
    fix = None
    if missing:
        fix = ("Λείπει: " + ", ".join(missing) + f". Python σε χρήση: {sys.executable}. Διπλό κλικ στο compare_embeddings.bat, "
               f"ή: \"{sys.executable}\" -m pip install -r requirements.txt")
    return {"ready": not missing, "missing": missing, "python": sys.executable, "fix": fix}


def evaluate(backend: str, data_dir, clients, repeats: int, log=print) -> Row:
    t0 = time.time()
    try:
        u = I.load_universe(data_dir)
        kad_ids = sorted(set(u["activity"]["activity_id"].dropna()))
        texts = [u["kad_desc"].get(k) or k for k in kad_ids]
        t1 = time.time()
        emb = E.embed_texts(texts, backend, force=True)       # real timing; the result is cached, so the model is loaded only once
        embed_s = time.time() - t1
        base = P.build_base(data_dir, backend)
    except Exception as e:                                    # model not downloadable, no memory, ...: report it and carry on
        return Row(backend, "skipped", {"reason": f"{type(e).__name__}: {e}"})
    build_s = round(time.time() - t0, 1)
    quality = kad_neighbour_quality(emb, kad_ids)
    log(f"{INFO.get(backend, (backend,))[0]}: embeddings έτοιμα σε {embed_s:.1f} δευτ., ποιότητα ελληνικών {100 * quality:.1f}%")
    M = HS._Market(base, clients, B=30)
    R.pu_features(base.model, base.companies["latitude"], base.companies["longitude"])

    def one(r):
        sp = HS.make_split(M.all_vats, feedback=0.0, test=0.2, salt=f"{HS.SALT}-r{r}")
        rp, rb = M.recall(M.rank(sp.train), sp.test), M.recall(M.baseline_rank(sp.train, r), sp.test)
        log(f"{INFO.get(backend, (backend,))[0]}: διαχωρισμός {r + 1}/{repeats}, Recall@500 {100 * rp[500]:.1f}%, Recall@2000 {100 * rp[2000]:.1f}%")
        return {**{f"product@{k}": rp[k] for k in C.KS}, **{f"baseline@{k}": rb[k] for k in C.KS}}
    r_all = HS._pmap(one, range(repeats))
    _, ana = M.analyse(M.all_vats)
    df = pd.DataFrame(r_all)
    return Row(backend, "done", {"dim": int(emb.shape[1]), "n_kad": len(texts), "build_s": build_s, "embed_s": round(embed_s, 2),
                                 "ms_per_kad": round(1000 * embed_s / len(texts), 2), "ms_per_business": round(1000 * embed_s / len(base.companies), 3),
                                 "peak_mb": peak_memory_mb(), "kad_neighbour_p5": round(quality, 4), "profiles": int(ana["meta"]["k"]), "per_repeat": df.to_dict("list"),
                                 **{c: round(float(df[c].mean()), 4) for c in df}})


def compare(rows: list[Row]) -> list[dict]:
    """Paired difference against the first finished backend on the same splits, with a 95% interval (normal approximation)."""
    done = [r for r in rows if r.status == "done"]
    if len(done) < 2:
        return []
    ref = done[0].detail["per_repeat"]
    out = []
    for r in done[1:]:
        row = {"backend": r.backend, "against": done[0].backend}
        for k in C.KS:
            d = np.array(r.detail["per_repeat"][f"product@{k}"]) - np.array(ref[f"product@{k}"])
            half = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            row[f"diff@{k}"] = round(float(d.mean()), 4); row[f"half@{k}"] = round(float(half), 4)
        out.append(row)
    return out


def verdict(cmp_row: dict) -> str:
    worse = any(cmp_row[f"diff@{k}"] + cmp_row[f"half@{k}"] < 0 for k in C.KS)
    better = any(cmp_row[f"diff@{k}"] - cmp_row[f"half@{k}"] > 0 for k in C.KS)
    return "Χειρότερο πέρα από τον θόρυβο" if worse and not better else "Καλύτερο πέρα από τον θόρυβο" if better and not worse else "Μεικτό" if better and worse else "Ισοδύναμο μέσα στον θόρυβο"


def to_markdown(rows: list[Row], cmp_rows: list[dict], repeats: int) -> str:
    p = lambda x: f"{100 * x:.1f}%"
    t = ["# Σύγκριση μοντέλων embeddings (Τεχνικό Brief, ενότητα 5.3)", "", f"Ίδια δεδομένα, ίδιοι {repeats} διαχωρισμοί 80/20 ανά ΑΦΜ, ίδιο pipeline.", "",
         "| Μοντέλο | Ποιότητα ελληνικών: ίδια κλάση ΚΑΔ στους 5 κοντινότερους | Recall@500 | Recall@2000 | Προφίλ | Διαστάσεις | Χρόνος embeddings (δ) | ms ανά επιχείρηση | Μέγιστη μνήμη (MB) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.status != "done":
            t.append(f"| {INFO.get(r.backend, (r.backend,))[0]} | δεν εκτελέστηκε: {r.detail['reason'][:90]} | | | | | | | |")
            continue
        d = r.detail
        t.append(f"| {INFO.get(r.backend, (r.backend,))[0]} | {p(d['kad_neighbour_p5'])} | {p(d['product@500'])} | {p(d['product@2000'])} | {d['profiles']} | {d['dim']} | {d['embed_s']} | {d['ms_per_business']} | {d['peak_mb']} |")
    done = [r for r in rows if r.status == "done"]
    if done:
        t += ["", f"Βάση αναφοράς: Recall@500 {p(done[0].detail['baseline@500'])}, Recall@2000 {p(done[0].detail['baseline@2000'])}. Η μέγιστη μνήμη είναι αθροιστική για όλη την εκτέλεση."]
    if cmp_rows:
        t += ["", "## Διαφορά έναντι " + cmp_rows[0]["against"] + " (ζευγαρωτά, 95%)", "", "| Μοντέλο | Δ Recall@500 | Δ Recall@2000 | Συμπέρασμα |", "|---|---|---|---|"]
        t += [f"| {INFO.get(c['backend'], (c['backend'],))[0]} | {100 * c['diff@500']:+.1f} ±{100 * c['half@500']:.1f} μον. | {100 * c['diff@2000']:+.1f} ±{100 * c['half@2000']:.1f} μον. | {verdict(c)} |" for c in cmp_rows]
    t += ["", "## Άδεια και προέλευση", "", "| Μοντέλο | Άδεια | Προέλευση |", "|---|---|---|"]
    t += [f"| {INFO[r.backend][0]} | {INFO[r.backend][1]} | {INFO[r.backend][2]} |" for r in rows if r.backend in INFO]
    t += ["", "Η άδεια πρέπει να επιβεβαιωθεί στο αρχείο LICENSE του συγκεκριμένου αποθετηρίου, όπως ζητά το brief.", ""]
    return "\n".join(t)


def insights(rows: list[Row], cmp_rows: list[dict]) -> list[dict]:
    done = [r for r in rows if r.status == "done"]
    out = []
    name = lambda b: INFO.get(b, (b,))[0]
    for r in rows:
        if r.status != "done":
            out.append({"level": "bad", "text": f"{name(r.backend)}: δεν εκτελέστηκε ({r.detail['reason'][:160]})."})
    for c in cmp_rows:
        out.append({"level": "info", "text": f"{name(c['backend'])} έναντι {name(c['against'])}: Recall@2000 {100 * c['diff@2000']:+.1f} μονάδες (±{100 * c['half@2000']:.1f}), Recall@500 {100 * c['diff@500']:+.1f} (±{100 * c['half@500']:.1f}). {verdict(c)}."})
    if done:
        q = max(done, key=lambda r: r.detail["kad_neighbour_p5"])
        out.append({"level": "info", "text": f"Καλύτερη ποιότητα στις ελληνικές περιγραφές ΚΑΔ: {name(q.backend)} ({100 * q.detail['kad_neighbour_p5']:.1f}%)."})
    active = next((c for c in cmp_rows if c["backend"] == C.DEFAULT_EMBED), None)
    if active and verdict(active).startswith("Χειρότερο"):
        out.append({"level": "warn", "text": f"Το μοντέλο της εφαρμογής ({name(C.DEFAULT_EMBED)}) βγαίνει χειρότερο από το {name(active['against'])}. Χρειάζεται απόφαση για το ποιο θα μείνει."})
    return out


def result_path(out_dir) -> Path:
    return Path(out_dir) / "embedding_comparison.json"


def load_result(out_dir) -> dict | None:
    try:
        return json.loads(result_path(out_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run(data_dir, clients, out_dir, models=None, repeats: int = 10, progress=lambda ev: None) -> dict:
    """The whole comparison: every model on the same splits. progress gets snapshots; the result is stored for the app and the brief check."""
    import threading
    models = list(models or DEFAULT_BACKENDS)
    t0, done, log, rows, lock = time.time(), [0], [], [], threading.Lock()
    total = len(models) * (1 + repeats)

    def snap(msg):
        el = time.time() - t0
        progress({"status": "running", "message": msg, "done": done[0], "total": total, "pct": round(100 * done[0] / total), "elapsed_s": round(el, 1),
                  "eta_s": None if done[0] == 0 else round(el / done[0] * (total - done[0]), 1), "log": log[-14:]})

    def tick(msg):
        with lock:
            done[0] += 1; log.append(msg); snap(msg)

    for i, b in enumerate(models):
        snap(f"{INFO.get(b, (b,))[0]}: φόρτωση μοντέλου και embeddings (η πρώτη φορά κατεβάζει το μοντέλο)")
        r = evaluate(b, data_dir, clients, repeats, log=tick)
        rows.append(r)
        with lock:
            done[0] = (i + 1) * (1 + repeats)
            if r.status != "done":
                log.append(f"{INFO.get(b, (b,))[0]}: δεν εκτελέστηκε"); snap(log[-1])
        gc.collect()
    cmp_rows = compare(rows)
    md = to_markdown(rows, cmp_rows, repeats)
    res = {"status": "done", "repeats": repeats, "models": [{"backend": r.backend, "name": INFO.get(r.backend, (r.backend,))[0], "status": r.status, "licence": INFO.get(r.backend, ("", "-", "-"))[1],
                                                               "origin": INFO.get(r.backend, ("", "-", "-"))[2], **r.detail} for r in rows],
           "comparison": cmp_rows, "verdicts": {c["backend"]: verdict(c) for c in cmp_rows}, "insights": insights(rows, cmp_rows),
           "neural_done": sum(r.status == "done" and r.backend.startswith("st:") for r in rows), "seconds": round(time.time() - t0, 1), "log": log}
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    (d / "embedding_comparison.md").write_text(md, encoding="utf-8")
    result_path(d).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res
