"""Feedback re-ranking: a sales person marks proposed businesses and the list learns, in about a second, on this computer.

Design rules
* The deliverable never changes. lookalikes.csv stays the model's ranking; feedback produces separate files.
* No labels = the model's ranking, exactly (tested).
* A label has a MEANING, not just a sign. "Wrong area" pushes down businesses near that one; it says nothing about their
  sector. "Closed" or "not now" removes one business and teaches nothing.
* Profiles stay as they are. Accepted businesses strengthen the profile they fit; feedback cannot silently redefine an ICP.
* Influence grows with evidence: alpha = strength * n / (n + PRIOR_LABELS). Two clicks cannot overturn the model.
* Fixed reason codes, no free text: nothing about solvency, payment behaviour or a person can be typed into the system.
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import config as C, lookalike as L, rankers as R

REASONS = {
    "good_fit":       {"label": "Ταιριάζει",                      "kind": "pos",  "group": "accept"},
    "already_client": {"label": "Είναι ήδη πελάτης",              "kind": "pos",  "group": "accept"},
    "wrong_sector":   {"label": "Λάθος κλάδος",                   "kind": "neg",  "group": "reject", "blocks": {"activity": 1.0}},
    "wrong_area":     {"label": "Εκτός περιοχής κάλυψης",         "kind": "neg",  "group": "reject", "blocks": {"geo": 0.7, "admin": 0.3}},
    "wrong_legal":    {"label": "Μη κατάλληλη νομική μορφή",      "kind": "neg",  "group": "reject", "blocks": {"legal": 1.0}},
    "not_fit":        {"label": "Δεν ταιριάζει (γενικά)",         "kind": "neg",  "group": "reject", "blocks": None},
    "closed":         {"label": "Δεν λειτουργεί πια",             "kind": "skip", "group": "skip"},
    "not_now":        {"label": "Όχι τώρα",                       "kind": "skip", "group": "skip"},
}
BLOCK_LABEL = {"activity": "δραστηριότητα", "geo": "τοποθεσία", "admin": "τοποθεσία", "legal": "νομική μορφή", "age": "ηλικία επιχείρησης", "struct": "δομή"}

PRIOR_LABELS = 12.0     # tuned on development clients only, never on the locked test set (scripts/hard_split.py)
POS_BOOST = 3.0         # an accepted business counts this many times the average client's weight
NEG_WEIGHT = 12.0       # weight of a rejected business in the classifier
REPEL = 0.30            # largest score penalty from targeted rejections
REPEL_POWER = 2.0
REPEL_PRIOR = 2.0
QUEUE_POOL = 1500


def tier(rank: int) -> str:
    """Priority band for people, using the two list lengths of the brief (section 8.1). A percentage of the market would suggest a
    precision the model does not have."""
    return "Πρώτες 500" if rank <= 500 else "Πρώτες 2.000" if rank <= 2000 else "Μετά τις 2.000"


def _avg_pct(x: np.ndarray) -> np.ndarray:
    """Percentile with ties sharing one value (the argsort-based percentile would order ties by row number)."""
    from scipy.stats import rankdata
    return (rankdata(x, method="average") - 1) / max(len(x) - 1, 1)


def repel_score(sim: np.ndarray) -> np.ndarray:
    """0..1: how much a business is 'like what was rejected', relative to the typical business, so that cosine similarities
    that are high across a whole sector, and one-hot similarities that are 0/1, are treated correctly."""
    med = min(float(np.median(sim)), 0.9)
    return np.clip((sim - med) / max(1.0 - med, 1e-6), 0.0, 1.0) ** REPEL_POWER


@dataclass
class Ranking:
    order: np.ndarray            # candidate rows, best first (labelled rows are not in it)
    score: np.ndarray            # same length, descending
    per_profile: np.ndarray      # n x k scores
    alpha: float
    gprofile: np.ndarray         # row -> index of the profile that selected it (-1 if not a candidate)


class FeedbackSession:
    """Labels of one analysis and the ranking they produce. Thread-safe; every change is appended to `history`."""

    def __init__(self, ctx, state: dict, Z: np.ndarray | None = None, strength: float = 1.0):
        self.ctx, self.state = ctx, state
        comp = ctx.companies
        self.Z = R.pu_features(ctx.model, comp["latitude"], comp["longitude"]) if Z is None else Z
        self.ids = comp["company_id"].astype(str).to_numpy()
        self.pos_of = {c: i for i, c in enumerate(self.ids)}
        self.base_rs = state["rank_score"]
        self.base_order = state["_default_order"]
        self.base_rank = np.full(len(comp), -1, np.int64); self.base_rank[self.base_order] = np.arange(1, len(self.base_order) + 1)
        self.is_cand = np.zeros(len(comp), bool); self.is_cand[state["cand"]] = True
        self.labels: dict[int, str] = {}
        self.history: list[dict] = []
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self._lock = threading.RLock()
        self._cache: Ranking | None = None

    # ------------------------------------------------------------------ labels
    def resolve(self, company_id) -> int:
        i = self.pos_of.get(str(company_id))
        if i is None:
            raise KeyError(f"unknown company {company_id}")
        if not self.is_cand[i]:
            raise ValueError("this business is one of the uploaded clients; only proposed businesses can be marked")
        return i

    def add(self, company_id, reason: str, source: str = "user") -> None:
        if reason not in REASONS:
            raise ValueError(f"unknown reason {reason}")
        with self._lock:
            i = self.resolve(company_id)
            self.history.append({"seq": len(self.history) + 1, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "action": "label",
                                 "company_id": self.ids[i], "reason": reason, "previous": self.labels.get(i), "source": source,
                                 "model_rank": int(self.base_rank[i])})
            self.labels[i] = reason
            self._cache = None

    def undo(self) -> dict | None:
        with self._lock:
            for h in reversed(self.history):
                if h["action"] == "label" and not h.get("undone"):
                    h["undone"] = True
                    i = self.pos_of[h["company_id"]]
                    if h["previous"] is None:
                        self.labels.pop(i, None)
                    else:
                        self.labels[i] = h["previous"]
                    self.history.append({"seq": len(self.history) + 1, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "action": "undo",
                                         "company_id": h["company_id"], "reason": h["reason"], "previous": None, "source": "user",
                                         "model_rank": h["model_rank"]})
                    self._cache = None
                    return h
        return None

    def reset(self) -> None:
        with self._lock:
            self.history.append({"seq": len(self.history) + 1, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "action": "reset",
                                 "company_id": "", "reason": "", "previous": None, "source": "user", "model_rank": None})
            self.labels.clear()
            self._cache = None

    def set_strength(self, s: float) -> None:
        with self._lock:
            s = float(np.clip(s, 0.0, 1.0))
            if s != self.strength:
                self.strength, self._cache = s, None

    def _rows(self, kind: str, labels: dict | None = None) -> np.ndarray:
        lab = self.labels if labels is None else labels
        return np.array(sorted(i for i, r in lab.items() if REASONS[r]["kind"] == kind), dtype=np.int64)

    def counts(self) -> dict:
        n = {g: 0 for g in ("accept", "reject", "skip")}
        for r in self.labels.values():
            n[REASONS[r]["group"]] += 1
        return n

    # ------------------------------------------------------------------ re-ranking
    def _per_profile(self, labels: dict) -> tuple[np.ndarray, float]:
        """n x k scores with the given labels. Starts from the stored model scores; returns (scores, alpha)."""
        pos, neg = self._rows("pos", labels), self._rows("neg", labels)
        n_teach = len(pos) + len(neg)
        rs = self.base_rs
        if n_teach == 0 or self.strength == 0.0:
            return rs, 0.0
        alpha = self.strength * n_teach / (n_teach + PRIOR_LABELS)
        model, Z = self.ctx.model, self.Z
        members = self.state["members"]
        home = rs[pos].argmax(axis=1) if len(pos) else np.empty(0, int)   # accepted businesses join the profile they fit best
        out = rs.copy()
        for c, (mpos, mw) in enumerate(members):
            add = pos[home == c]
            allp = np.concatenate([mpos, add])
            allw = np.concatenate([mw, np.full(len(add), POS_BOOST * float(np.mean(mw)))]) if len(add) else mw
            knn, _ = L.profile_scores(model, allp, allw, C.KNN_M, C.CENTROID_BLEND)
            pu = R.pu_scores(Z, allp, allw / allw.mean(), neg=neg if len(neg) else None, neg_w=NEG_WEIGHT)
            out[:, c] = (1 - alpha) * rs[:, c] + alpha * (C.RANK_W_KNN * R._pct(knn) + C.RANK_W_PU * R._pct(pu))
        for reason, spec in REASONS.items():                              # targeted rejections
            if spec["kind"] != "neg" or spec.get("blocks") is None:
                continue
            rows = np.array(sorted(i for i, r in labels.items() if r == reason), dtype=np.int64)
            if len(rows) == 0:
                continue
            m = min(3, len(rows))
            sim = model.sim_to(rows, spec["blocks"])
            near = np.partition(sim, -m, axis=1)[:, -m:].mean(axis=1) if len(rows) > m else sim.max(axis=1)
            out -= (REPEL * self.strength * len(rows) / (len(rows) + REPEL_PRIOR) * repel_score(near))[:, None]
        return out, alpha

    def _build(self, labels: dict) -> Ranking:
        cand = self.state["cand"]
        rs, alpha = self._per_profile(labels)
        share = self.state["share"]
        # merged over ALL candidates, marked rows are taken out afterwards: removing one business must not reorder the others
        g = R.merge_proportional(rs[cand], share)
        o = np.argsort(-g, kind="stable")
        pr = np.stack([np.argsort(np.argsort(-rs[cand, c], kind="stable"), kind="stable") for c in range(rs.shape[1])], axis=1)
        gp = np.full(len(rs), -1, np.int64)
        gp[cand] = np.argmin(pr / np.maximum(share, 1e-9)[None, :], axis=1)
        if labels:
            o = o[~np.isin(cand[o], np.array(sorted(labels), dtype=np.int64))]
        return Ranking(cand[o], g[o], rs, alpha, gp)

    def ranking(self) -> Ranking:
        with self._lock:
            if self._cache is None:
                self._cache = self._build(self.labels)
            return self._cache

    def impact(self, before: np.ndarray, top: int = 100) -> dict:
        """What the last change did to the list, in plain counts (the order before the change is passed in)."""
        now = self.ranking().order
        marked = np.array(sorted(self.labels), dtype=np.int64)
        before = before[~np.isin(before, marked)]                # a row that was just marked leaves the list: that is not a move of the others
        old, new = set(before[:top].tolist()), set(now[:top].tolist())
        pos = np.full(len(self.base_rank), -1, np.int64); pos[before] = np.arange(len(before))
        risers = [(int(pos[i]) - r, int(i)) for r, i in enumerate(now[:top]) if pos[i] >= 0]
        gain, i = max(risers, default=(0, -1))
        return {"top": top, "entered": len(new - old), "left": len(old - new), "top_riser": str(self.ctx.companies["company_name_el"].iat[i]) if gain > 0 else None, "top_riser_moved": max(gain, 0)}

    # ------------------------------------------------------------------ explaining a move
    def explain(self, i: int, moved: int) -> str:
        """One deterministic Greek sentence: the labelled business this one resembles most, and in which respect."""
        pos, neg = self._rows("pos"), self._rows("neg")
        sign_rows = (pos, "εγκρίνατε") if moved > 0 else (neg, "απορρίψατε")
        rows, verb = sign_rows
        if abs(moved) == 0 or len(rows) == 0:
            return "Δεν επηρεάστηκε ουσιαστικά από τις επισημάνσεις σας."
        model = self.ctx.model
        blocks = ("activity", "geo", "legal", "age")
        best = None
        for b in blocks:
            X = model.blocks[b]
            s = X[i] @ X[rows].T
            s = np.asarray(s.todense() if hasattr(s, "todense") else s).ravel()
            typical = float(np.mean(s))
            j = int(s.argmax())
            contrast = float(s[j]) * model.weights.get(b, 0.0) + (float(s[j]) - typical) * 0.25
            if best is None or contrast > best[0]:
                best = (contrast, b, int(rows[j]))
        name = str(self.ctx.companies["company_name_el"].iat[best[2]] or "")
        return f"{'Ανέβηκε' if moved > 0 else 'Κατέβηκε'} {abs(moved)} θέσεις: μοιάζει ως προς {BLOCK_LABEL[best[1]]} με «{name[:50]}» που {verb}."

    # ------------------------------------------------------------------ what to review next
    def queue(self, n: int = 12, explore: float = 0.0) -> list[dict]:
        """Next businesses to review: the top of the list. Optionally a share (`explore`) is chosen far from what has already been
        shown (greedy max-min distance). Off by default: on development clients it did not beat plain top-of-list (0.67 vs 0.70)."""
        rk = self.ranking()
        n = max(1, min(int(n), 100))
        pool = rk.order[:QUEUE_POOL]
        if len(pool) == 0:
            return []
        n_ex = int(round(n * explore)) if len(pool) > n else 0
        top = list(pool[:n - n_ex])
        tags = {int(i): "top" for i in top}
        if n_ex:
            Z = self.Z[pool].astype(np.float32); Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
            lab = np.array(sorted(self.labels), dtype=np.int64)
            refs = np.vstack([Z[:len(top)], self._unit(self.Z[lab])]) if len(lab) else Z[:len(top)]
            d = (Z @ refs.T).max(axis=1)
            d[:len(top)] = np.inf
            for _ in range(n_ex):
                j = int(np.argmin(d - 0.15 * (1 - np.arange(len(pool)) / len(pool))))      # far from the shown set, still high on the list
                tags[int(pool[j])] = "explore"
                top.append(pool[j])
                d = np.maximum(d, Z @ Z[j]); d[j] = np.inf
        return [{"pos": int(i), "why": tags[int(i)]} for i in top]

    @staticmethod
    def _unit(Z):
        Z = Z.astype(np.float32)
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)

    # ------------------------------------------------------------------ does the feedback generalise?
    def validate(self, folds: int = 5, seed: int = C.SEED) -> dict:
        """Honest self-check on the person's own labels: hide a fifth of them, learn from the rest, and test whether the
        learned ranking separates the hidden accepted from the hidden rejected better than the model alone did."""
        with self._lock:
            lab = dict(self.labels)
        rows = np.array(sorted(i for i, r in lab.items() if REASONS[r]["kind"] != "skip"), dtype=np.int64)
        y = np.array([REASONS[lab[int(i)]]["kind"] == "pos" for i in rows], dtype=bool)
        if y.sum() < 4 or (~y).sum() < 4:
            return {"status": "too_few", "accepted": int(y.sum()), "rejected": int((~y).sum()), "needed_each": 4}
        rng = np.random.default_rng(seed)
        fold = np.empty(len(rows), int)
        for cls in (True, False):                                         # stratified
            idx = rng.permutation(np.flatnonzero(y == cls)); fold[idx] = np.arange(len(idx)) % min(folds, int(min(y.sum(), (~y).sum())))
        share = self.state["share"]
        keys_model, keys_fb, out_y = [], [], []
        for f in np.unique(fold):
            te = rows[fold == f]
            train_lab = {int(i): lab[int(i)] for i in rows[fold != f]}
            rs, _ = self._per_profile(train_lab)
            weighted = lambda M: (M * share[None, :]).sum(axis=1)
            keys_model.append(weighted(self.base_rs)[te]); keys_fb.append(weighted(rs)[te]); out_y.append(y[fold == f])
            # scores are compared inside the fold only, where every row saw the same model
            keys_model[-1] = _avg_pct(keys_model[-1]); keys_fb[-1] = _avg_pct(keys_fb[-1])
        m, b, yy = np.concatenate(keys_model), np.concatenate(keys_fb), np.concatenate(out_y)
        auc = lambda s: float(((s[yy][:, None] > s[~yy][None, :]).mean() + 0.5 * (s[yy][:, None] == s[~yy][None, :]).mean()))
        am, ab = auc(m), auc(b)
        return {"status": "done", "accepted": int(y.sum()), "rejected": int((~y).sum()), "auc_model": round(am, 3), "auc_feedback": round(ab, 3),
                "helps": bool(ab > am), "note": "Έλεγχος μόνο στις δικές σας επισημάνσεις: κρύβεται ένα μέρος τους, το σύστημα μαθαίνει από τις υπόλοιπες και μετράμε αν ξεχωρίζει τις κρυμμένες."}

    # ------------------------------------------------------------------ exports
    def log_frame(self) -> pd.DataFrame:
        cols = ["seq", "ts", "action", "company_id", "reason", "previous", "source", "model_rank", "undone"]
        with self._lock:
            return pd.DataFrame(self.history, columns=cols)

    def export_frame(self, top: int = 5000) -> pd.DataFrame:
        """Accepted businesses first, then the re-ranked list. Rejected/skipped ones are not in it."""
        rk = self.ranking()
        acc = sorted((i for i, r in self.labels.items() if r == "good_fit"), key=lambda i: self.base_rank[i])
        rows = list(acc) + list(rk.order[:max(0, top - len(acc))])
        comp = self.ctx.companies
        pid = np.array(self.state["profile_ids"])
        best_prof = self.base_rs[rows].argmax(axis=1) if rows else np.empty(0, int)
        prof = [pid[rk.gprofile[i]] if rk.gprofile[i] >= 0 else pid[best_prof[n]] for n, i in enumerate(rows)]
        return pd.DataFrame({"rank": np.arange(1, len(rows) + 1), "company_id": self.ids[rows], "name": comp["company_name_el"].to_numpy()[rows],
                             "municipality": comp["municipality_name"].to_numpy()[rows], "profile_id": prof,
                             "status": ["accepted" if int(i) in self.labels else "proposed" for i in rows],
                             "model_rank": self.base_rank[rows]})
