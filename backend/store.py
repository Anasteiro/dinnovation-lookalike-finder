"""Small relational store (SQLite, part of Python: no server, no install, works offline) for runs and results.
Every analysis is saved with its settings and timestamp, so any number shown in the app can be traced back to the run
that produced it (AI Act traceability) and results survive a restart. Heavy maths stays in NumPy; SQL is the persistence
and query layer (the universe tables can be loaded through it as well)."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, created_at TEXT, source TEXT, k INTEGER, n_clients INTEGER, matched_vats INTEGER,
  universe INTEGER, selection_rule TEXT, params_json TEXT, timings_json TEXT);
CREATE TABLE IF NOT EXISTS profiles(run_id TEXT, profile_id TEXT, title TEXT, n_clients INTEGER, turnover_share REAL, stability REAL,
  confidence REAL, description TEXT, description_is_ai INTEGER, features_json TEXT, PRIMARY KEY(run_id, profile_id));
CREATE TABLE IF NOT EXISTS assignments(run_id TEXT, company_id TEXT, profile_id TEXT, score REAL);
CREATE TABLE IF NOT EXISTS lookalikes(run_id TEXT, profile_id TEXT, company_id TEXT, rank INTEGER, score REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS benchmarks(run_id TEXT PRIMARY KEY, created_at TEXT, result_json TEXT);
CREATE INDEX IF NOT EXISTS ix_look ON lookalikes(run_id, profile_id, rank);
CREATE INDEX IF NOT EXISTS ix_assign ON assignments(run_id, company_id);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.executescript(SCHEMA)

    def _c(self):
        con = sqlite3.connect(self.path, timeout=30)
        return con

    def save_run(self, run_id: str, source: str, res: dict):
        meta = res["meta"]
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        con = self._c()
        try:
            with con:
                con.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
                for t in ("profiles", "assignments", "lookalikes"):
                    con.execute(f"DELETE FROM {t} WHERE run_id=?", (run_id,))
                con.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (run_id, now, source, meta["k"], meta["match"].get("client_vats"), meta["match"].get("matched_vats"),
                             meta["universe"], meta["selection_rule"], json.dumps(meta["params"], ensure_ascii=False, default=str),
                             json.dumps(meta["timings"])))
                con.executemany("INSERT INTO profiles VALUES(?,?,?,?,?,?,?,?,?,?)",
                                [(run_id, p["profile_id"], p["title"], p["n_clients"], p["turnover_share"], p["stability"], p["confidence"],
                                  p["description"], int(bool(p["description_is_ai"])), json.dumps(p["features"], ensure_ascii=False))
                                 for p in res["profiles"]])
                a = res["assignments"]
                con.executemany("INSERT INTO assignments VALUES(?,?,?,?)",
                                [(run_id, str(c), p, float(s)) for c, p, s in zip(a["company_id"], a["profile_id"], a["score"])])
                l = res["lookalikes"]
                con.executemany("INSERT INTO lookalikes VALUES(?,?,?,?,?,?)",
                                [(run_id, p, str(c), int(r), float(s), t) for p, c, r, s, t in
                                 zip(l["profile_id"], l["company_id"], l["rank"], l["score"], l["αιτιολόγηση"])])
        finally:
            con.close()

    def update_description(self, run_id: str, profile_id: str, text: str, is_ai: bool):
        con = self._c()
        try:
            with con:
                con.execute("UPDATE profiles SET description=?, description_is_ai=? WHERE run_id=? AND profile_id=?",
                            (text, int(is_ai), run_id, profile_id))
        finally:
            con.close()

    def save_benchmark(self, run_id: str, result: dict):
        con = self._c()
        try:
            with con:
                con.execute("INSERT OR REPLACE INTO benchmarks VALUES(?,?,?)",
                            (run_id, time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(result, ensure_ascii=False)))
        finally:
            con.close()

    def runs(self, limit=50):
        con = self._c()
        try:
            cur = con.execute("SELECT run_id, created_at, source, k, n_clients, matched_vats, universe FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            con.close()
