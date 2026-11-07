"""Part 7: local web API (FastAPI). Runs on localhost only; no external service is ever called.

Flow:  POST /api/analyze  ->  job_id  ->  GET /api/jobs/{id}/events (SSE progress)  ->  GET /api/results/{id}
       GET /api/lookalikes, /api/geo/points, /api/transparency/{id}, POST /api/jobs/{id}/narrate, GET /api/export/...
The universe (embeddings, blocks) is built once and cached; each upload only re-runs the client-dependent part.
"""
from __future__ import annotations
import json
import tempfile
import threading
import time
import uuid
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import chat as CH
from . import rankers as R
from . import analysis as A, benchmark as BM, clarity as K, external as X, enrich as EN, whatif as WI, config as C, ingest as I, llm as LLM, pipeline as P, store as ST

MAX_UPLOAD = 5 * 1024 * 1024
EXPORTS = {"profiles.json": "application/json", "assignments.csv": "text/csv", "lookalikes.csv": "text/csv",
           "lookalikes_global.csv": "text/csv", "assignments_detail.csv": "text/csv", "analysis_meta.json": "application/json",
           "activity_clarity.csv": "text/csv", "refined_lookalikes.csv": "text/csv",
           "lookalikes_by_profile.csv": "text/csv"}

LICENSES = [
    {"item": "Στοιχεία επιχειρήσεων", "source": "ΓΕΜΗ Open Data (opendata.businessportal.gr)", "license": "ODC-BY 1.0"},
    {"item": "Στίγματα GPS", "source": "© OpenStreetMap contributors (γεωκωδικοποίηση)", "license": "ODbL"},
    {"item": "EuroLLM-9B-Instruct", "source": "Unbabel, IST, EU-funded", "license": "Apache-2.0"},
    {"item": "Πληθυσμός δήμων (Απογραφή 2021, προσωρινά)", "source": "ΕΛΣΤΑΤ - αντίγραφο github.com/tdiam/greece-population-census-2021", "license": "CC0 1.0"},
    {"item": "Όρια Ελλάδας για τον χάρτη", "source": "Natural Earth 1:10m (μέσω world-atlas)", "license": "Δημόσιο κτήμα"},
]
AI_ACT = [
    {"q": "Τι αποφασίζει ή προτείνει το σύστημα;",
     "a": "Προτείνει ποιες επιχειρήσεις μοιάζουν με τους καλύτερους πελάτες μιας εταιρείας (B2B πωλήσεις). Δεν παίρνει αυτόματες αποφάσεις και δεν παράγει βαθμολογία φερεγγυότητας."},
    {"q": "Ποιος επηρεάζεται από αυτό;",
     "a": "Οι επιχειρήσεις του dataset (μεταξύ τους ατομικές επιχειρήσεις) μπορεί να λάβουν εμπορική προσέγγιση. Δεν υπάρχει επίπτωση σε πίστωση, απασχόληση ή δικαιώματα."},
    {"q": "Τι βλέπει ο χρήστης για το πώς προέκυψε το αποτέλεσμα;",
     "a": "Στατιστικά με lift, p/q-value, σταθερότητα και βεβαιότητα αντιστοίχισης ανά profile· αιτιολόγηση ανά lookalike· σήμανση AI στα κείμενα και έλεγχος groundedness."},
    {"q": "Πού μπορεί να κάνει λάθος και πώς το εντοπίζουμε;",
     "a": "Οι ΚΑΔ είναι δηλωτικοί και θορυβώδεις· το LLM μπορεί να διατυπώσει λάθος. Μετράμε Recall@k με cross-validation, ελέγχουμε σταθερότητα με bootstrap και επαληθεύουμε αυτόματα κάθε αριθμό του κειμένου."},
]


class Job:
    def __init__(self, source: str):
        self.id = uuid.uuid4().hex[:12]
        self.source, self.created = source, time.time()
        self.status, self.error, self.events = "queued", None, []
        self.result, self.ctx, self.company_table = None, None, None
        self.narration = {"status": "idle"}
        self.benchmark = {"status": "idle"}

    def log(self, step, **kw):
        self.events.append({"t": round(time.time() - self.created, 2), "step": step, **kw})


def _json(o):
    return json.loads(json.dumps(o, default=A._py, ensure_ascii=False))


def _records(df: pd.DataFrame):
    return json.loads(df.to_json(orient="records", force_ascii=False))


def company_table(ctx) -> pd.DataFrame:
    c = ctx.companies
    return pd.DataFrame({
        "company_id": c["company_id"].values, "name": c["company_name_el"].fillna("").values,
        "municipality": c["municipality_name"].values, "prefecture": c["prefecture_name"].values,
        "legal_type": c["legal_type"].values, "primary_kad": ctx.primary_kad,
        "primary_kad_label": [ctx.kad_desc.get(k, "") for k in ctx.primary_kad],
        "founded": c["incorporation_date"].dt.year.values, "lat": c["latitude"].values, "lon": c["longitude"].values,
        **({k: ctx.factors[k].values for k in ctx.factors.columns} if getattr(ctx, "factors", None) is not None else {}),
        **({"clarity": ctx.clarity["clarity"].values, "clarity_level": ctx.clarity["level"].values,
            "declared_other_kads": ctx.clarity["n_other"].values, "unrelated_kads": ctx.clarity["n_unrelated"].values}
           if getattr(ctx, "clarity", None) is not None else {}),
    }).set_index("company_id")


def create_app(data_dir: Path | None = None, embed: str | None = None, llm_model: str | None = None,
               out_root: Path | None = None, llm_factory=None, frontend_dir: Path | None = None,
               auto_narrate: bool = True) -> FastAPI:
    import os
    data_dir = Path(data_dir or os.environ.get("FORCECRM_DATA") or C.DATA_DIR)
    embed = embed or os.environ.get("FORCECRM_EMBED", "tfidf")
    llm_model = llm_model or os.environ.get("FORCECRM_LLM")
    out_root = Path(out_root or C.OUT_DIR / "jobs")
    custom_llm = llm_factory is not None                       # tests / alternative back-ends
    llm_factory = llm_factory or (lambda model: LLM.ollama_generate(model))
    frontend_dir = Path(frontend_dir or C.ROOT / "frontend" / "dist")

    app = FastAPI(title="Local ICP Engine", version="0.4")
    # Only pages served from this computer may call the API (blocks other websites from reading results through the
    # browser). FORCECRM_ALLOW_ORIGINS="*" re-opens it, e.g. for a hosted design preview.
    origins = os.environ.get("FORCECRM_ALLOW_ORIGINS")
    cors = dict(allow_methods=["*"], allow_headers=["*"])
    if origins:
        cors["allow_origins"] = [o.strip() for o in origins.split(",")]
    else:
        cors["allow_origin_regex"] = r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"
    try:
        app.add_middleware(CORSMiddleware, allow_private_network=True, **cors)
    except TypeError:                                         # older Starlette
        app.add_middleware(CORSMiddleware, **cors)

    state = {"base": None, "error": None}
    store = ST.Store(out_root / "runs.db")
    _enr = {"mtime": None, "db": {"places": {}, "fetched": {}, "stats": {}}}

    def enrichment():
        f = Path(data_dir or C.DATA_DIR) / "enrichment" / "places.json"
        m = f.stat().st_mtime if f.exists() else None
        if m != _enr["mtime"]:
            _enr["mtime"], _enr["db"] = m, EN.load(f) if m else {"places": {}, "fetched": {}, "stats": {}}
        return _enr["db"]
    jobs: dict[str, Job] = {}
    build_lock, run_lock = threading.Lock(), threading.Lock()

    def get_base():
        with build_lock:
            if state["base"] is None:
                t = time.time()
                state["base"] = P.build_base(data_dir, embed)
                b = state["base"]                     # client-independent features: computed once, reused by every analysis
                R.pu_features(b.model, b.companies["latitude"], b.companies["longitude"])
                state["build_s"] = round(time.time() - t, 1)
            return state["base"]
    app.state.warmup = get_base

    def job_or_404(jid) -> Job:
        j = jobs.get(jid)
        if not j:
            raise HTTPException(404, "unknown job")
        return j

    def done_job(jid) -> Job:
        j = job_or_404(jid)
        if j.status == "error":
            raise HTTPException(422, j.error)
        if j.status != "done":
            raise HTTPException(409, f"job is {j.status}")
        return j

    def start_narration(j: Job, model: str) -> str:
        """Writes the Greek descriptions in the background (local Ollama). Returns the narration status."""
        if j.narration["status"] == "running":
            return "running"
        j.narration = {"status": "running", "model": model, "done": 0, "total": len(j.result["profiles"])}

        def work():
            try:
                gen = llm_factory(model)

                def prog(i, n, r):
                    j.narration.update(done=i)
                    j.log("narrated", profile=r["profile_id"], verified=r["verified"], source=r["source"], attempts=r["attempts"])
                LLM.narrate_profiles(j.result["profiles"], gen, C.CACHE_DIR / "llm", prog)
                (out_root / j.id / "profiles.json").write_text(json.dumps(j.result["profiles"], ensure_ascii=False, indent=2, default=A._py), encoding="utf-8")
                for p in j.result["profiles"]:
                    try:
                        store.update_description(j.id, p["profile_id"], p["description"], p.get("description_is_ai", False))
                    except Exception:
                        pass
                j.narration["status"] = "done"
                j.log("narration_done")
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                j.narration = {"status": "error", "message": f"Ollama not reachable / model missing: {e}"}
                j.log("narration_error", message=j.narration["message"])
            except Exception as e:
                j.narration = {"status": "error", "message": str(e)}
                j.log("narration_error", message=str(e))
        threading.Thread(target=work, daemon=True).start()
        return "running"

    def ollama_up() -> bool:
        if custom_llm:
            return True
        try:
            import urllib.request
            urllib.request.urlopen(LLM.OLLAMA_URL + "/api/tags", timeout=1).read()
            return True
        except Exception:
            return False

    def run_job(job: Job, clients: pd.DataFrame, k, boot, match):
        with run_lock:
            try:
                job.status = "running"
                job.log("loading_market_data")
                base = get_base()
                job.log("matching_clients")
                ctx = P.attach_clients(base, clients, match)
                job.log("matched", **ctx.match_stats)
                if ctx.match_stats["matched_vats"] < 8:
                    raise ValueError(f"Ταυτοποιήθηκαν μόνο {ctx.match_stats['matched_vats']} ΑΦΜ στη βάση· χρειάζονται τουλάχιστον 8. "
                                     f"Only {ctx.match_stats['matched_vats']} VATs matched the dataset; at least 8 are needed.")
                res = A.run_analysis(ctx, None, k=k, B=boot, progress=lambda s, **kw: job.log(s, **kw))
                job.ctx, job.result, job.company_table = ctx, res, company_table(ctx)
                A.write_outputs(res, out_root / job.id)
                try:
                    store.save_run(job.id, job.source, res)
                except Exception as e:             # the database is a convenience layer: never fail an analysis because of it
                    job.log("store_warning", message=str(e))
                job.status = "done"
                job.log("done", total_s=res["meta"]["timings"]["total_s"])
                if auto_narrate and llm_model and ollama_up():       # AI descriptions arrive a few seconds after the results
                    start_narration(job, llm_model)
            except FileNotFoundError as e:
                job.status, job.error = "error", f"Data files not found ({e}). Run `python run.py unlock` first."
                job.log("error", message=job.error)
            except Exception as e:                                  # surfaced to the UI, never swallowed
                job.status, job.error = "error", str(e)
                job.log("error", message=job.error)

    # ------------------------------------------------------------------ endpoints
    @app.get("/api/health")
    def health():
        ollama = ollama_up()
        has_data = any(data_dir.rglob("vwHackCompany.csv")) if data_dir.exists() else False
        return {"status": "ok", "data_found": has_data, "market_data_loaded": state["base"] is not None,
                "embedding": embed, "ollama_running": ollama, "llm_model": llm_model, "jobs": len(jobs)}

    @app.post("/api/analyze")
    async def analyze(file: UploadFile | None = File(None), use_sample: bool = Form(False),
                      k: int | None = Form(None), boot: int = Form(30), match: str = Form("split")):
        if match not in ("split", "head", "full"):
            raise HTTPException(400, "match must be split|head|full")
        try:
            if use_sample:
                path, name = I.find_file(data_dir, "*Pelatologio*.xlsx"), "sample client list"
                clients = I.load_clients(path)
            else:
                if file is None:
                    raise HTTPException(400, "Upload a file (xlsx/csv) or set use_sample=true")
                blob = await file.read()
                if len(blob) > MAX_UPLOAD:
                    raise HTTPException(413, "file too large (max 5 MB)")
                suffix = Path(file.filename or "x.csv").suffix.lower()
                if suffix not in (".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".txt"):
                    raise HTTPException(400, "Unsupported file type; use .xlsx or .csv")
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(blob)
                try:
                    clients = I.load_clients(Path(tmp.name))
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
                name = file.filename
        except FileNotFoundError as e:
            raise HTTPException(503, f"Data files not found: {e}")
        except ValueError as e:
            raise HTTPException(400, str(e))
        job = Job(name)
        jobs[job.id] = job
        job.log("uploaded", client_vats=int(len(clients)), file=name)
        threading.Thread(target=run_job, args=(job, clients, k, max(0, min(boot, 100)), match), daemon=True).start()
        return {"job_id": job.id}

    @app.get("/api/jobs/{jid}")
    def job_status(jid: str):
        j = job_or_404(jid)
        return {"job_id": j.id, "status": j.status, "error": j.error, "events": j.events, "narration": j.narration, "benchmark": j.benchmark}

    @app.post("/api/jobs/{jid}/benchmark")
    def start_benchmark(jid: str, repeats: int = Form(8)):
        """Live accuracy check on the uploaded clients (hidden 20% vs baseline). Runs in the background, poll GET."""
        j = done_job(jid)
        if j.benchmark["status"] == "running":
            return j.benchmark
        reps = max(2, min(int(repeats), 30))
        j.benchmark = {"status": "running", "repeats": reps}

        def work():
            try:
                with run_lock:
                    res = BM.run_benchmark(j.ctx, k=j.result["meta"]["k"], repeats=reps)
                j.benchmark = res
                try:
                    store.save_benchmark(j.id, res)
                except Exception:
                    pass
            except Exception as e:
                j.benchmark = {"status": "error", "message": str(e)}
        threading.Thread(target=work, daemon=True).start()
        return j.benchmark

    @app.get("/api/jobs/{jid}/benchmark")
    def get_benchmark(jid: str):
        return job_or_404(jid).benchmark

    @app.get("/api/runs")
    def runs():
        return {"runs": store.runs()}

    @app.get("/api/jobs/{jid}/events")
    def job_events(jid: str):
        j = job_or_404(jid)

        def gen():
            i = 0
            while True:
                while i < len(j.events):
                    yield f"data: {json.dumps(j.events[i], ensure_ascii=False)}\n\n"
                    i += 1
                if j.status in ("done", "error") and i >= len(j.events):         # results first; narration is polled separately
                    break
                time.sleep(0.15)
            yield f"event: end\ndata: {json.dumps({'status': j.status})}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/results/{jid}")
    def results(jid: str):
        j = done_job(jid)
        m = j.result["meta"]
        dataset = {"market_size": m["universe"], "candidates": m["candidates"], "client_vats": m["match"]["client_vats"],
                   "matched_vats": m["match"]["matched_vats"], "unmatched_vats": m["match"]["unmatched_vats"],
                   "client_shops": m["match"]["client_shops"], "profiles_found": m["k"],
                   "lookalike_rows": int(len(j.result["lookalikes"])), "none_share": m["none_share"],
                   "execution_seconds": m["timings"]["total_s"], "source": j.source}
        return _json({"job_id": j.id, "dataset": dataset, "meta": m, "profiles": j.result["profiles"]})

    @app.get("/api/lookalikes")
    def lookalikes(job: str, profile: str = "all", limit: int = 50, offset: int = 0, q: str = ""):
        j = done_job(job)
        res = j.result
        if profile == "all":
            df = res["lookalikes_global"][["rank", "company_id", "profile_id", "score", "αιτιολόγηση"]].rename(columns={"αιτιολόγηση": "reason"})
        else:
            df = res["lookalikes"][res["lookalikes"]["profile_id"] == profile].rename(columns={"αιτιολόγηση": "reason"})
            if df.empty:
                raise HTTPException(404, "unknown profile")
        df = df.join(j.company_table, on="company_id")
        n_cand = max(len(res["_state"]["cand"]), 1) if "_state" in res else max(len(j.company_table), 1)
        df = df.assign(top_pct=(df["rank"] / n_cand).round(5))     # position in the market: rank 1 of 24k -> top 0.004%
        if q.strip():
            # plain-text (never regex: user text as a regex could crash or hang the server), case- and accent-insensitive
            ql = CH.norm(q.strip()[:80])
            m = False
            for col in ("name", "municipality", "primary_kad_label"):
                m = m | df[col].fillna("").astype(str).map(CH.norm).str.contains(ql, regex=False)
            df = df[m]
        limit = max(1, min(limit, 500))
        page = df.iloc[offset: offset + limit]
        items = _records(page)
        places = enrichment()["places"]
        for it in items:                                   # optional extra info (Google Places), null when not fetched / not matched
            it["google"] = places.get(str(it.get("company_id")))
        return {"total": int(len(df)), "offset": offset, "limit": limit, "items": items}

    @app.get("/api/enrichment/status")
    def enrichment_status():
        db = enrichment()
        dates = sorted({v for v in db["fetched"].values() if v})
        return {"available": bool(db["places"]), "source": EN.SOURCE, "fetched_on": dates[-1] if dates else None,
                "note": "Πρόσθετες πληροφορίες από το Google Places, ανακτημένες εκ των προτέρων και αποθηκευμένες τοπικά. Δεν επηρεάζουν τη βαθμολόγηση.",
                **{k: db["stats"].get(k) for k in ("looked_up", "matched", "match_rate")}}

    @app.get("/api/enrichment/places")
    def enrichment_places(job: str, limit: int = 300):
        """Pins for the map: enriched lookalikes (in global rank order) with their location and Google details."""
        j = done_job(job)
        places = enrichment()["places"]
        g = j.result["lookalikes_global"].head(5000)
        t = j.company_table
        out = []
        for r in g.itertuples():
            pl = places.get(str(r.company_id))
            if pl and str(r.company_id) in t.index.astype(str):
                row = t.loc[r.company_id]
                out.append({"rank": int(r.rank), "company_id": str(r.company_id), "name": str(row["name"] or ""), "lat": None if pd.isna(row["lat"]) else float(row["lat"]),
                            "lon": None if pd.isna(row["lon"]) else float(row["lon"]), "google": pl})
            if len(out) >= max(1, min(limit, 1000)):
                break
        return {"source": EN.SOURCE, "items": out}

    @app.get("/api/geo/points")
    def geo_points(job: str, kind: str = "lookalikes", profile: str = "all", limit: int = 500):
        j = done_job(job)
        limit = max(1, min(limit, 3000))
        if kind == "clients":
            ids = j.ctx.client_shops["company_id"]
            det = j.result["assignments_detail"].set_index("company_id")["profile_id"]
            df = pd.DataFrame({"company_id": ids.values, "profile_id": ids.map(det).values})
        else:
            df = (j.result["lookalikes_global"] if profile == "all" else
                  j.result["lookalikes"][j.result["lookalikes"]["profile_id"] == profile])[["company_id", "profile_id", "score"]].head(limit)
        df = df.join(j.company_table[["name", "municipality", "lat", "lon"]], on="company_id").dropna(subset=["lat", "lon"])
        return {"kind": kind, "items": _records(df.head(limit))}

    @app.get("/api/geo/density")
    def geo_density(job: str, kind: str = "lookalikes", profile: str = "all", cell: float = 0.1, limit: int = 2000):
        """Offline map data: counts per grid cell (cell size in degrees). kind = market | clients | lookalikes (top `limit`)."""
        j = done_job(job)
        cell = max(0.01, min(float(cell), 1.0))
        t = j.company_table
        if kind == "market":
            ids = t.index
        elif kind == "clients":
            ids = j.ctx.client_shops["company_id"]
            if profile != "all":
                det = j.result["assignments_detail"].set_index("company_id")["profile_id"]
                ids = ids[ids.map(det).values == profile]
        else:
            g = j.result["lookalikes_global"] if profile == "all" else j.result["lookalikes"][j.result["lookalikes"]["profile_id"] == profile]
            ids = g["company_id"].head(max(1, min(limit, 5000)))
        d = t.loc[t.index.intersection(pd.Index(ids))][["lat", "lon"]].dropna()
        cells = d.assign(la=(d.lat // cell) * cell + cell / 2, lo=(d.lon // cell) * cell + cell / 2).groupby(["la", "lo"]).size().reset_index(name="n")
        return {"kind": kind, "profile": profile, "cell_deg": cell, "total": int(len(d)),
                "bounds": {"lat": [float(t.lat.min()), float(t.lat.max())], "lon": [float(t.lon.min()), float(t.lon.max())]},
                "cells": [{"lat": round(float(r.la), 4), "lon": round(float(r.lo), 4), "n": int(r.n)} for r in cells.itertuples()]}

    @app.get("/api/factors")
    def factors(job: str):
        """Catalogue for the filter page: external factors (with source and range), filter options and the profiles' factor medians."""
        j = done_job(job)
        c = j.ctx.companies
        cat = j.ctx.factor_catalog or []
        yrs = c["incorporation_date"].dt.year.dropna()
        return {"factors": cat,
                "options": {"legal": [{"value": str(k), "count": int(v)} for k, v in c["legal_type"].value_counts().items()],
                            "prefectures": sorted(str(x) for x in c["prefecture_name"].dropna().unique()),
                            "founded": [int(yrs.min()), int(yrs.max())] if len(yrs) else None, "branches": int(c["is_branch"].fillna(0).astype(int).sum())},
                "profiles": [{"profile_id": p["profile_id"], "title": p["title"], "external_factors": p.get("external_factors", [])} for p in j.result["profiles"]],
                "note": "Τα εξωτερικά δεδομένα είναι τοπικά αρχεία (καμία κλήση στο διαδίκτυο). Τα φίλτρα και οι προτιμήσεις αλλάζουν μόνο τη λίστα που βλέπετε· τα παραδοτέα αρχεία μένουν ίδια."}

    @app.post("/api/jobs/{jid}/refine")
    def refine(jid: str, body: dict):
        """Filter + extra factors on top of the model ranking.
        body: {profile: "all"|id, filters: {legal[], prefectures[], min_clarity, founded_from, founded_to, exclude_branches, factors{key:{min,max}}},
               boosts: {factor_key: -1..1}, strength: 0..1 (default 0.5), limit, offset}"""
        j = done_job(jid)
        st = j.result["_state"]
        prof = body.get("profile", "all")
        if prof == "all":
            order, score = st["_default_order"], st["_default_score"]
        else:
            if prof not in st["profile_ids"]:
                raise HTTPException(404, "unknown profile")
            c = st["profile_ids"].index(prof)
            sc = st["rank_score"][st["cand"], c]
            o = np.argsort(-sc, kind="stable")
            order, score = st["cand"][o], sc[o]
        strength = float(min(1.0, max(0.0, body.get("strength", 0.5))))
        pos, sc, removed = X.refine(order, score, j.ctx.companies, j.ctx.factors, j.ctx.clarity, body.get("filters") or {}, body.get("boosts") or {}, strength)
        base_rank = np.empty(len(j.ctx.companies), int); base_rank[order] = np.arange(1, len(order) + 1)
        t = j.company_table
        ids = j.ctx.companies["company_id"].to_numpy()[pos]
        df = t.loc[ids].reset_index()
        df.insert(0, "rank", np.arange(1, len(df) + 1)); df.insert(1, "model_rank", base_rank[pos]); df["refined_score"] = np.round(sc, 5)
        df["profile_id"] = prof
        out_cols = ["rank", "model_rank", "company_id", "name", "municipality", "prefecture", "legal_type", "primary_kad_label", "founded",
                    "clarity", "clarity_level"] + [k for k in j.ctx.factors.columns] + ["refined_score"]
        df[[c for c in out_cols if c in df.columns]].head(5000).to_csv(out_root / j.id / "refined_lookalikes.csv", index=False, encoding="utf-8")
        lim = max(1, min(int(body.get("limit", 50)), 500)); off = max(0, int(body.get("offset", 0)))
        page = df.iloc[off: off + lim]
        top = df.head(2000)
        summary = {k: (None if top[k].isna().all() else round(float(top[k].median()), 3)) for k in j.ctx.factors.columns}
        lab = {c["key"]: c["label"] for c in (j.ctx.factor_catalog or [])}
        removed = {lab.get(k, k): v for k, v in removed.items()}
        return {"total": int(len(df)), "removed": removed, "offset": off, "limit": lim, "items": _records(page),
                "top2000_medians": summary, "export": "refined_lookalikes.csv"}

    @app.get("/api/clarity")
    def clarity_info(job: str, company_id: str | None = None):
        """Activity Clarity Index: market distribution, per-profile summary, or one business with its declared codes."""
        j = done_job(job)
        cl = j.ctx.clarity
        if cl is None:
            raise HTTPException(404, "clarity not available")
        if company_id is not None:
            if company_id not in cl.index.astype(str):
                raise HTTPException(404, "unknown company")
            key = cl.index[cl.index.astype(str) == company_id][0]
            r = cl.loc[key]
            a = j.ctx.activity[j.ctx.activity["company_id"] == key][["activity_id", "atype"]].drop_duplicates()
            codes = cl.attrs.get("codes")
            relmap = {}
            if codes is not None:
                cc = codes[codes["company_id"] == key]
                relmap = dict(zip(cc["activity_id"].astype(str), cc["relevance"]))
            kads = []
            for x in a.itertuples():
                code = str(x.activity_id)
                rel = 1.0 if x.atype == "primary" else relmap.get(code)
                kads.append({"code": code, "type": x.atype, "description": j.ctx.kad_desc.get(code, ""),
                             "related": None if rel is None else bool(rel >= 0.5)})
            kads.sort(key=lambda d: (d["type"] != "primary", d["related"] is not False, d["code"]))
            return {"company_id": company_id, "clarity": int(r["clarity"]), "level": r["level"], "declared_other_kads": int(r["n_other"]),
                    "unrelated_kads": int(r["n_unrelated"]), "explanation": K.explain(r), "kads": kads[:80], "kads_total": len(kads)}
        lv = cl["level"].value_counts(normalize=True)
        hist = np.histogram(cl["clarity"], bins=[0, 20, 40, 50, 60, 80, 90, 101])[0].tolist()
        return {"method": K.__doc__.strip().splitlines()[0], "levels": {k: round(float(lv.get(k, 0.0)), 3) for k in ("Καθαρό", "Μέτριο", "Θορυβώδες")},
                "histogram": {"edges": [0, 20, 40, 50, 60, 80, 90, 100], "counts": hist}, "market_mean": round(float(cl["clarity"].mean()), 1),
                "profiles": {p["profile_id"]: p.get("activity_clarity") for p in j.result["profiles"]},
                "note": "Ο δείκτης περιγράφει την ποιότητα της δήλωσης ΚΑΔ (πόσο καθαρή είναι η δηλωμένη δραστηριότητα), όχι την επιχείρηση. Δεν επηρεάζει την κατάταξη."}

    @app.get("/api/geo/outline")
    def geo_outline():
        """Land outline of Greece (GeoJSON Feature, [lon, lat]) bundled with the app: Natural Earth 1:10m, public domain. Works offline."""
        import json as _json
        f = Path(__file__).parent / "assets" / "greece.geojson"
        return _json.loads(f.read_text(encoding="utf-8"))

    @app.get("/api/geo/municipalities")
    def geo_municipalities(job: str, profile: str = "all", limit: int = 12, top: int = 2000):
        """Where the best lookalikes are: municipalities ranked by how many of the top-`top` lookalikes they hold (with their share of the market)."""
        j = done_job(job)
        t = j.company_table
        g = j.result["lookalikes_global"] if profile == "all" else j.result["lookalikes"][j.result["lookalikes"]["profile_id"] == profile]
        sel = t.loc[t.index.intersection(pd.Index(g["company_id"].head(max(1, min(top, 5000)))))]
        cnt = sel.groupby("municipality").size().sort_values(ascending=False).head(max(1, min(limit, 50)))
        mk = t.groupby("municipality").size()
        loc = t.groupby("municipality")[["lat", "lon"]].mean()
        items = [{"municipality": str(m), "lookalikes": int(n), "market": int(mk[m]), "share_of_market_in_top": round(float(n / mk[m]), 3),
                  "lat": None if pd.isna(loc.loc[m, "lat"]) else round(float(loc.loc[m, "lat"]), 4),
                  "lon": None if pd.isna(loc.loc[m, "lon"]) else round(float(loc.loc[m, "lon"]), 4)} for m, n in cnt.items()]
        return {"profile": profile, "top": top, "items": items}

    @app.post("/api/jobs/{jid}/whatif")
    def whatif(jid: str, activity: float = Form(1.0), geo: float = Form(1.0), legal: float = Form(1.0), age: float = Form(1.0),
               evaluate: bool = Form(False), top: int = Form(25)):
        """Re-rank with different importance for activity / location / legal form / age (multipliers 0-2, 1 = product setting).
        With evaluate=true it also measures Recall@500/2000 for that setting on the hidden-client benchmark (about 6 s)."""
        j = done_job(jid)
        mult = {"activity": activity, "geo": geo, "legal": legal, "age": age}
        with run_lock:
            out = WI.run(j.ctx, j.result["_state"], mult, top=max(5, min(top, 100)))
            if evaluate:
                b = BM.run_benchmark(j.ctx, k=j.result["meta"]["k"], repeats=4, mult=mult)
                out["evaluation"] = {"methods": b["methods"], "lift_vs_baseline": b["lift_vs_baseline"], "repeats": b["repeats"], "note": b["note"]}
        return out

    assistants: dict[str, CH.Assistant] = {}

    @app.post("/api/jobs/{jid}/chat")
    def chat(jid: str, body: dict):
        """Grounded read-only assistant. body: {question: str (<=400 chars), ai_summary: bool}.
        Answers only from this job's results; the question is never sent to a language model."""
        j = done_job(jid)
        a = assistants.get(jid)
        if a is None:
            a = assistants[jid] = CH.Assistant(j.result, j.company_table, j.ctx)
        if not a.rate_ok():
            raise HTTPException(429, "Πάρα πολλές ερωτήσεις. Δοκιμάστε ξανά σε λίγο.")
        q = body.get("question") if isinstance(body, dict) else None
        if not isinstance(q, str):
            raise HTTPException(400, "question must be text")
        t0 = time.time()
        out = a.answer(q)
        out["ai_summary"] = None
        if body.get("ai_summary") and out.get("facts") and llm_model:
            try:
                gen = llm_factory(llm_model)
                txt = CH.llm_summary(out["facts"], gen)
            except Exception:
                txt = None
            out["ai_summary"] = ({"text": txt, "label": f"Κείμενο AI ({llm_model}, τοπικά) - ελεγμένο ότι δεν προσθέτει αριθμούς"}
                                 if txt else {"text": None, "label": "Η σύνοψη AI δεν πέρασε τον έλεγχο ή το τοπικό μοντέλο δεν είναι διαθέσιμο."})
        out["question"] = CH.clean_input(q)
        out["llm_available"] = bool(llm_model)
        out["seconds"] = round(time.time() - t0, 3)
        out.pop("facts", None)
        return _json(out)

    @app.post("/api/jobs/{jid}/narrate")
    def narrate(jid: str, model: str | None = Form(None)):
        j = done_job(jid)
        model = model or llm_model
        if not model:
            raise HTTPException(400, "No LLM model set. Start the server with --llm-model <ollama name> or pass model.")
        return {"status": start_narration(j, model)}

    @app.get("/api/transparency/{jid}")
    def transparency(jid: str):
        j = done_job(jid)
        m = j.result["meta"]
        return _json({
            "execution": {"timings_s": m["timings"], "market_data_build_s": state.get("build_s"), "all_local": True,
                          "external_ai_calls": 0},
            "models": {"embedding": embed, "llm": llm_model, "llm_options": LLM.GEN_OPTIONS,
                       "clustering": "σταθμισμένο kernel k-means πάνω στην ίδια ομοιότητα (cosine ανά ομάδα χαρακτηριστικών) που χρησιμοποιείται για ανάθεση και κατάταξη",
                       "statistics": "μονόπλευρος υπεργεωμετρικός έλεγχος (Fisher) ανά χαρακτηριστικό έναντι μη-πελατών, διόρθωση Benjamini-Hochberg, shrunk lift"},
            "selection": {"rule": m["selection_rule"], "k_table": m["k_table"], "params": m["params"]},
            "profiles": [{"profile_id": p["profile_id"], "stability": p["stability"], "confidence": p["confidence"],
                          "description_is_ai": p["description_is_ai"], "description_meta": p.get("description_meta")}
                         for p in j.result["profiles"]],
            "licenses": LICENSES, "ai_act": AI_ACT,
            "limits": ["Οι ΚΑΔ είναι δηλωτικοί και θορυβώδεις.", "Ο έλεγχος groundedness ελέγχει παραπομπές και αριθμούς, όχι κάθε νοηματική λεπτομέρεια.",
                       "Τα αποτελέσματα δείχνουν ομοιότητα, όχι πρόβλεψη αγοράς ή φερεγγυότητας."]})

    @app.get("/api/export/{jid}/{name}")
    def export(jid: str, name: str):
        j = done_job(jid)
        if name not in EXPORTS:
            raise HTTPException(404, "unknown export")
        path = out_root / j.id / name
        if not path.exists():
            raise HTTPException(404, "file not written")
        return FileResponse(path, media_type=EXPORTS[name], filename=name)

    # ------------------------------------------------------------------ front-end (static build or built-in console)
    if (frontend_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    else:
        console = (Path(__file__).parent / "console.html").read_text(encoding="utf-8")

        @app.get("/", response_class=HTMLResponse)
        def home():
            return console
    return app
