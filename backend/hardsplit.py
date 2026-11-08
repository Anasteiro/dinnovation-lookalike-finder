"""Hard split: a leakage-proof partition of the client list, and the evaluation protocols built on it.

Why this exists. A plain repeated 80/20 split answers "how good is the ranker on average", but every weight in the product
was chosen while looking at those same clients. A hard split adds three guarantees:

1. PARTITION BY HASH.  Each VAT goes to exactly one part (train / feedback / locked test) from sha256(salt|VAT). It does not
   depend on file order, on the random state or on which other clients are in the file, and it can be re-derived by anyone.
2. ONE-WAY INFORMATION.  Ranking and feedback code only ever receive train (and feedback-oracle) rows. `assert_disjoint`
   runs on every evaluation, on VATs not rows, so sibling shops of one VAT can never be on two sides.
3. THE ORACLE IS BLIND TO THE TEST SET.  In the feedback simulation a locked-test client that appears in the review list
   gets exactly the answer any other non-client gets. Skipping it would itself leak its identity.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import numpy as np
import pandas as pd

SALT = "forcecrm-hardsplit-v1"


class LeakageError(AssertionError):
    """Information from a held-out set reached a place it must not reach."""


def vat_bucket(vat: str, salt: str = SALT) -> float:
    """Uniform number in [0, 1) that depends only on the VAT and the salt."""
    return int(hashlib.sha256(f"{salt}|{vat}".encode()).hexdigest()[:13], 16) / 16 ** 13


@dataclass(frozen=True)
class HardSplit:
    train: frozenset
    feedback: frozenset
    test: frozenset
    salt: str

    def fingerprint(self) -> str:
        """Short id of this exact partition: printed next to every number so a result can be traced to its split."""
        h = hashlib.sha256()
        for name, part in (("train", self.train), ("feedback", self.feedback), ("test", self.test)):
            h.update(name.encode() + b":" + ",".join(sorted(part)).encode() + b";")
        return h.hexdigest()[:12]

    def sizes(self) -> dict:
        return {"train": len(self.train), "feedback": len(self.feedback), "test": len(self.test)}


def assert_disjoint(*parts) -> None:
    seen: set = set()
    for p in parts:
        p = set(p)
        both = seen & p
        if both:
            raise LeakageError(f"{len(both)} VAT(s) on two sides of the split, e.g. {sorted(both)[:3]}")
        seen |= p


def make_split(vats, feedback: float = 0.2, test: float = 0.2, salt: str = SALT) -> HardSplit:
    """Deterministic partition by VAT hash. The cut points are exact quantiles of the hashes, so sizes are exact."""
    uniq = sorted({str(v) for v in vats})
    if feedback < 0 or test <= 0 or feedback + test >= 1:
        raise ValueError("need 0 <= feedback, 0 < test, feedback + test < 1")
    order = sorted(uniq, key=lambda v: (vat_bucket(v, salt), v))
    n_test, n_fb = int(round(test * len(uniq))), int(round(feedback * len(uniq)))
    te, fb, tr = order[:n_test], order[n_test:n_test + n_fb], order[n_test + n_fb:]
    assert_disjoint(tr, fb, te)
    if len(tr) + len(fb) + len(te) != len(uniq):
        raise LeakageError("the partition lost or duplicated a VAT")
    return HardSplit(frozenset(tr), frozenset(fb), frozenset(te), salt)


# ---------------------------------------------------------------------------------------------------- protocols
import os                                                # noqa: E402
import threading                                         # noqa: E402
import time                                              # noqa: E402
from concurrent.futures import ThreadPoolExecutor        # noqa: E402
from . import analysis as A, config as C, evalharness as H, ingest as I, pipeline as P, rankers as R  # noqa: E402

WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1 or 1))
REGIONS = {
    "Αττική": ["ΑΤΤΙΚΗΣ", "ΑΘΗΝΩΝ", "ΠΕΙΡΑΙΩΣ", "ΑΝΑΤΟΛΙΚΗΣ ΑΤΤΙΚΗΣ", "ΔΥΤΙΚΗΣ ΑΤΤΙΚΗΣ"],
    "Κρήτη": ["ΗΡΑΚΛΕΙΟΥ", "ΧΑΝΙΩΝ", "ΡΕΘΥΜΝΗΣ", "ΛΑΣΙΘΙΟΥ"],
    "Νησιά Αιγαίου": ["ΚΥΚΛΑΔΩΝ", "ΔΩΔΕΚΑΝΗΣΟΥ", "ΛΕΣΒΟΥ", "ΣΑΜΟΥ", "ΧΙΟΥ"],
    "Κεντρική Μακεδονία": ["ΘΕΣΣΑΛΟΝΙΚΗΣ", "ΧΑΛΚΙΔΙΚΗΣ", "ΠΙΕΡΙΑΣ", "ΣΕΡΡΩΝ", "ΗΜΑΘΙΑΣ", "ΠΕΛΛΗΣ", "ΚΙΛΚΙΣ"],
}


class _Market:
    """Everything a hard-split protocol needs, computed once. Rankers only ever see the `train` clients handed to `.rank`."""

    def __init__(self, base, clients, B):
        self.base, self.clients, self.B = base, clients.drop_duplicates("vat").reset_index(drop=True), B
        self.vat = base.companies["vat"].to_numpy()
        self.n = len(self.vat)
        self.in_market = set(self.vat)
        self.clients = self.clients[self.clients["vat"].isin(self.in_market)].reset_index(drop=True)
        self.all_vats = self.clients["vat"].tolist()
        self._memo: dict = {}                               # the same training set is never ranked twice

    def analyse(self, train_vats, full=False, fresh=False):
        key = (frozenset(train_vats), full, id(self.clients))
        if not fresh and key in self._memo:
            return self._memo[key]
        ctx = P.attach_clients(self.base, self.clients[self.clients["vat"].isin(set(train_vats))])
        res = ctx, A.run_analysis(ctx, B=self.B, rank_only=not full)
        if not fresh:
            self._memo[key] = res
        return res

    def rank(self, train_vats, fresh=False):
        """Rank (0 = best) of every row with the product's method; train rows get a huge rank."""
        _, out = self.analyse(train_vats, fresh=fresh)
        return self._rank_of(out["_state"]["_default_order"])

    def _rank_of(self, order):
        r = np.full(self.n, 10 ** 9, np.int64); r[order] = np.arange(len(order))
        return r

    def baseline_rank(self, train_vats, seed=0):
        tr = np.isin(self.vat, list(train_vats))
        sc = H.baseline_scorer(self.base.primary_kad)(np.flatnonzero(tr), None)
        return self._rank_of(H.rank_order(sc, np.flatnonzero(tr), np.random.default_rng(seed))[:int((~tr).sum())])

    def recall(self, rank, test_vats, ks=C.KS) -> dict:
        """VAT-level: a hidden client is found if any of its shops is in the top k."""
        pos = np.flatnonzero(np.isin(self.vat, list(test_vats)))
        best = pd.Series(rank[pos]).groupby(self.vat[pos]).min().to_numpy()
        return {k: float((best < k).mean()) for k in ks}


def _holdout(M: _Market, train, test, name: str, note: str = "") -> dict:
    assert_disjoint(train, test)
    if len(train) < 20 or len(test) < 3:
        return {"name": name, "skipped": f"train={len(train)}, test={len(test)}"}
    rp, rb = M.recall(M.rank(train), test), M.recall(M.baseline_rank(train), test)
    chance = {k: k / max(M.n - len(train), 1) for k in C.KS}
    return {"name": name, "note": note, "n_train": len(train), "n_test": len(test),
            **{f"product@{k}": round(rp[k], 4) for k in C.KS}, **{f"baseline@{k}": round(rb[k], 4) for k in C.KS},
            **{f"chance@{k}": round(chance[k], 4) for k in C.KS}}


def _bootstrap(hits: np.ndarray, B=2000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    means = hits[rng.integers(0, len(hits), (B, len(hits)))].mean(axis=1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def locked_test(M: _Market, split: HardSplit) -> dict:
    """The one number: model built on train+feedback (80%), scored once on the locked 20%, bootstrap CI over the hidden VATs."""
    train = split.train | split.feedback
    assert_disjoint(train, split.test)
    out = _holdout(M, train, split.test, "Κλειδωμένο τεστ 20%")
    rp, rb = M.rank(train), M.baseline_rank(train)
    pos = np.flatnonzero(np.isin(M.vat, list(split.test)))
    for k in C.KS:
        out[f"product@{k}_ci95"] = _bootstrap((rp[pos] < k).astype(float))
        out[f"baseline@{k}_ci95"] = _bootstrap((rb[pos] < k).astype(float))
    out["fingerprint"] = split.fingerprint()
    return out


def repeated(M: _Market, repeats: int, tick=lambda *a: None) -> dict:
    """Same protocol under independent hash salts: the spread shows how much one locked split could mislead."""
    def one(r):
        sp_ = make_split(M.all_vats, feedback=0.0, test=0.2, salt=f"{SALT}-r{r}")
        rp, rb = M.recall(M.rank(sp_.train), sp_.test), M.recall(M.baseline_rank(sp_.train, r), sp_.test)
        tick(f"Διαχωρισμός {r + 1}: Recall@500 {100 * rp[500]:.1f}%, Recall@2000 {100 * rp[2000]:.1f}%")
        return {**{f"product@{k}": rp[k] for k in C.KS}, **{f"baseline@{k}": rb[k] for k in C.KS}}
    df = pd.DataFrame(_pmap(one, range(repeats)))
    return {"repeats": repeats, **{c: round(float(df[c].mean()), 4) for c in df}, **{c + "_sd": round(float(df[c].std(ddof=1)), 4) for c in df if repeats > 1},
            **{f"min@{k}": round(float(df[f"product@{k}"].min()), 4) for k in C.KS}, **{f"max@{k}": round(float(df[f"product@{k}"].max()), 4) for k in C.KS},
            **{f"wins@{k}": int((df[f"product@{k}"] > df[f"baseline@{k}"]).sum()) for k in C.KS}}


def generalisation_plan(M: _Market) -> list:
    """Splits that are hard on purpose: everything of one kind is hidden, so the model must reach a place, sector or period
    it has never seen. Random 80/20 cannot show this. Returns [(name, note, hidden VATs)]."""
    comp = M.base.companies.drop_duplicates("vat").set_index("vat")
    v = pd.Series(M.all_vats, index=M.all_vats)
    pref = comp["prefecture_name"].reindex(v.index)
    plan = [(f"Άγνωστη περιοχή: {name}", "Όλοι οι πελάτες της περιοχής κρυμμένοι", set(v[pref.isin(prefs)])) for name, prefs in REGIONS.items()]
    pkad = pd.Series(M.base.primary_kad, index=M.base.companies["vat"].to_numpy())
    pkad = pkad[~pkad.index.duplicated()].reindex(v.index).astype(str).str[:4]
    plan += [(f"Άγνωστος κλάδος: ΚΑΔ {code}", "Όλοι οι πελάτες του κλάδου κρυμμένοι", set(v[pkad == code])) for code, n in pkad.value_counts().items() if n >= 25]
    yr = comp["incorporation_date"].dt.year.reindex(v.index)
    plan.append(("Χρονικό: μάθηση ≤2022, τεστ 2023-24", "Οι νεότερες επιχειρήσεις κρυμμένες", set(v[yr >= 2023])))
    rev = M.clients.set_index("vat")["revenue"].reindex(v.index)
    if rev.notna().sum() >= 20:
        plan.append(("Μεγάλοι πελάτες: κρυμμένο το κορυφαίο 20% τζίρου", "Μάθηση μόνο από μικρότερους πελάτες", set(v[rev >= rev.quantile(0.8)])))
    _, ana = M.analyse(M.all_vats, full=True)
    profiles = ana["assignments_detail"].set_index("company_id")["profile_id"]
    s = P.attach_clients(M.base, M.clients).client_shops.assign(p=lambda d: d["company_id"].map(profiles))
    for p, g in s[s["p"] != "none"].groupby("p"):
        plan.append((f"Άγνωστο προφίλ: {p}", "Ολόκληρο το ICP κρυμμένο", set(g["vat"]) - set(s.loc[s["p"] != p, "vat"])))
    return plan


def hard_generalisation(M: _Market, plan=None, tick=lambda *a: None, on_row=lambda r: None) -> list[dict]:
    plan = generalisation_plan(M) if plan is None else plan
    all_v = set(M.all_vats)
    def one(item):
        name, note, te = item
        r = _holdout(M, all_v - te, te, name, note)
        tick(f"{name}: " + ("παραλείφθηκε" if "skipped" in r else f"Recall@2000 {100 * r['product@2000']:.1f}%"))
        on_row(r)
        return r
    return _pmap(one, plan)


def canaries(M: _Market, split: HardSplit, seed=C.SEED, tick=lambda *a: None, quick=False) -> list[dict]:
    """Tests of the evaluation itself. Each returns passed=True/False. quick: the three that cost at most one ranking."""
    train = sorted(split.train | split.feedback)
    out, rng = [], np.random.default_rng(seed)
    def add(row):
        out.append(row); tick(("Πέρασε: " if row["passed"] else "ΑΠΕΤΥΧΕ: ") + row["name"])
    # 1 random clients must score like chance: if not, something other than the client list drives the ranking
    fake = rng.choice(sorted(M.in_market), size=len(M.all_vats), replace=False).tolist()
    fsp = make_split(fake, feedback=0.0, test=0.2, salt=SALT + "-fake")
    saved = M.clients
    M.clients = pd.DataFrame({"vat": fake, "revenue": np.nan})
    try:
        r = M.recall(M.rank(fsp.train), fsp.test)
    finally:
        M.clients = saved
    ch = 2000 / (M.n - len(fsp.train))
    sd = np.sqrt(ch * (1 - ch) / len(fsp.test))
    add({"name": "Τυχαίοι «πελάτες» → επίπεδο τύχης", "observed": round(r[2000], 4), "expected": round(ch, 4), "passed": bool(abs(r[2000] - ch) < 4 * sd)})
    # 2 the same training clients with a different hidden set must give the same ranking, bit for bit
    if not quick:
        alt = make_split(train, feedback=0.0, test=0.2, salt=SALT + "-alt")
        a, b = M.rank(alt.train, fresh=True), M.rank(sorted(alt.train), fresh=True)
        add({"name": "Η κατάταξη δεν αλλάζει με το τι είναι κρυμμένο", "passed": bool(np.array_equal(a, b))})
    # 3 market features must be identical before and after a client list is attached
    h0 = _blocks_hash(M.base.model)
    M.analyse(train)
    add({"name": "Τα χαρακτηριστικά αγοράς δεν εξαρτώνται από τους πελάτες", "passed": h0 == _blocks_hash(M.base.model)})
    if not quick:
        # 4 positive control: plant a real leak and check that the harness sees it. A harness that cannot see a leak proves nothing
        Z = R.pu_features(M.base.model, M.base.companies["latitude"], M.base.companies["longitude"])
        leak = np.isin(M.vat, list(split.test | split.train | split.feedback)).astype(np.float32)[:, None] * 8.0
        M.base.model._pu_features = np.hstack([Z, leak])
        try:
            rl = M.recall(M.rank(train, fresh=True), split.test)
        finally:
            M.base.model._pu_features = Z
        honest = M.recall(M.rank(train), split.test)
        add({"name": "Θετικός έλεγχος: φυτεμένη διαρροή ανιχνεύεται", "observed": round(rl[500], 4), "expected": f"πολύ πάνω από {honest[500]:.2f}", "passed": bool(rl[500] > honest[500] + 0.3)})
    # 5 no train VAT can be proposed back
    ctx, ana = M.analyse(train)
    prop = set(ctx.companies["vat"].to_numpy()[ana["_state"]["_default_order"]])
    add({"name": "Κανένας πελάτης εκπαίδευσης δεν προτείνεται", "passed": not (prop & set(train))})
    return out


def _pmap(fn, items):
    """Parallel map, results in input order. The heavy work is NumPy, so threads share one read-only market in memory."""
    items = list(items)
    workers = min(WORKERS, len(items))
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def _blocks_hash(model) -> str:
    h = hashlib.sha256()
    for name in sorted(model.blocks):
        m = model.blocks[name]
        data = getattr(m, "data", m)
        h.update(name.encode()); h.update(np.asarray(data).tobytes())
    return h.hexdigest()[:16]


def feedback_blind(M: _Market, split: HardSplit, batch=50, rounds=3) -> dict:
    """Blind-oracle simulation. The oracle knows only the FEEDBACK clients; everybody else, locked-test clients included, gets
    the same answer. Compared on what is left in the list, so both arms lose the same reviewed rows.
    Non-clients are not truly bad fits, so this is the hardest setting for feedback and it mainly shows that it does no harm."""
    from . import feedback as FB
    assert_disjoint(split.train, split.feedback, split.test)
    ctx, ana = M.analyse(split.train)
    st = ana["_state"]
    s = FB.FeedbackSession(ctx, st)
    fb = set(split.feedback)
    found = 0
    for _ in range(rounds):
        for x in s.queue(batch, explore=0.0):
            i = x["pos"]
            if M.vat[i] in fb:
                s.add(s.ids[i], "good_fit"); found += 1
            else:
                s.add(s.ids[i], "not_fit")
    rm = np.array(sorted(s.labels), dtype=np.int64)
    te = np.flatnonzero(np.isin(M.vat, list(split.test)))
    te = te[~np.isin(te, rm)]
    base_o = st["_default_order"]; base_o = base_o[~np.isin(base_o, rm)]
    rb, rf = M._rank_of(base_o), M._rank_of(s.ranking().order)
    return {"name": "Τυφλός χρήστης", "reviewed": len(s.labels), "accepted": found, "test_clients_left": int(len(te)),
            **{f"without@{k}": round(float((rb[te] < k).mean()), 4) for k in C.KS}, **{f"with@{k}": round(float((rf[te] < k).mean()), 4) for k in C.KS}}


def intent_concepts(base) -> dict:
    c = base.companies
    pk = pd.Series(base.primary_kad).astype(str).str[:4]
    pr, lg = c["prefecture_name"], c["legal_type"]
    return {"Μόνο Κρήτη": (pr.isin(REGIONS["Κρήτη"]), "wrong_area"), "Μόνο Αττική": (pr.isin(REGIONS["Αττική"]), "wrong_area"),
            "Μόνο νησιά Αιγαίου": (pr.isin(REGIONS["Νησιά Αιγαίου"]), "wrong_area"), "Μόνο ΙΚΕ": (lg == "ΙΚΕ", "wrong_legal"),
            "Μόνο καφέ/μπαρ (5630)": (pk == "5630", "wrong_sector"), "Μόνο εστιατόρια (5611)": (pk == "5611", "wrong_sector")}


def intent_adaptation(M: _Market, split: HardSplit, batch=20, rounds=2, tick=lambda *a: None, generic=True, only=None) -> list[dict]:
    """A simulated sales person with one clear intent marks 40 businesses, giving the matching reason. Measured: the share of the
    next 100 proposals that match the intent, against the same list without feedback, and against reason-less rejection.
    This measures how fast the list adapts to the user, not client recall."""
    from . import feedback as FB
    ctx, ana = M.analyse(split.train | split.feedback)
    st = ana["_state"]
    items = list(intent_concepts(M.base).items())[:only]
    def one(item):
        name, (mask, code) = item
        m = np.asarray(mask, bool)
        row = {"intent": name, "labels": batch * rounds}
        for tag, reason in (("reasons", code), ("generic", "not_fit"))[:2 if generic else 1]:
            s = FB.FeedbackSession(ctx, st)
            for _ in range(rounds):
                for x in s.queue(batch, explore=0.0):
                    i = x["pos"]; s.add(s.ids[i], "good_fit" if m[i] else reason)
            rm = np.array(sorted(s.labels), dtype=np.int64)
            base_o = st["_default_order"]; base_o = base_o[~np.isin(base_o, rm)]
            if tag == "reasons":
                row["model_p100"] = round(float(m[base_o[:100]].mean()), 3)
            row[f"{tag}_p100"] = round(float(m[s.ranking().order[:100]].mean()), 3)
        tick(f"{name}: από {100 * row['model_p100']:.0f}% σε {100 * row['reasons_p100']:.0f}%")
        return row
    return _pmap(one, items)


def _ks(x):
    return f"{100 * x:.1f}".replace(".", ",") + "%"


def insights(r: dict) -> list[dict]:
    """Plain-language conclusions for a sales person. Computed from the numbers on screen, only for sections already finished."""
    out = []
    L, R_ = r.get("locked"), r.get("repeated")
    if L:
        t = L["n_test"]; f = int(round(L["product@2000"] * t)); b = int(round(L["baseline@2000"] * t))
        out.append({"level": "good" if f > b else "warn", "text": f"Από {t} πελάτες που κρύψαμε, οι {f} βρέθηκαν στις πρώτες 2.000 προτάσεις. Η απλή μέθοδος του ίδιου ΚΑΔ βρήκε {b}."})
    if R_:
        out.append({"level": "good" if R_["wins@2000"] == R_["repeats"] else "warn", "text": f"Σε {R_['repeats']} διαφορετικούς διαχωρισμούς ξεπεράσαμε τη βάση {R_['wins@2000']} φορές στις 2.000. Μέσος όρος {_ks(R_['product@2000'])}, από {_ks(R_['min@2000'])} έως {_ks(R_['max@2000'])}."})
    if L and R_ and R_["repeats"] > 1 and abs(L["product@2000"] - R_["product@2000"]) > R_.get("product@2000_sd", 1):
        out.append({"level": "info", "text": "Το κλειδωμένο τεστ απέχει από τον μέσο όρο: με λίγους κρυμμένους πελάτες ένας μόνο διαχωρισμός μετακινείται πολύ. Πιο αντιπροσωπευτικός είναι ο μέσος όρος."})
    gen = [g for g in (r.get("generalisation") or []) if "skipped" not in g]
    cold = [g for g in gen if g["name"].startswith(("Άγνωστη", "Άγνωστος", "Άγνωστο"))]
    if cold:
        if max(g["product@2000"] for g in cold) <= max(g["chance@2000"] for g in cold) * 1.5:
            out.append({"level": "warn", "text": "Όταν κρύβουμε όλους τους πελάτες μιας περιοχής, ενός κλάδου ή ενός προφίλ, η απόδοση πέφτει στο επίπεδο της τύχης. Το σύστημα βρίσκει όμοιους με όσους ήδη ξέρει· δεν ανακαλύπτει αγορές που λείπουν από το πελατολόγιο."})
    c = r.get("canaries")
    if c:
        ok = sum(x["passed"] for x in c)
        out.append({"level": "good" if ok == len(c) else "bad", "text": f"Έλεγχοι διαρροής: πέρασαν {ok} από {len(c)}."})
    fb = (r.get("feedback_blind") or [None])[0]
    if fb:
        out.append({"level": "info", "text": f"Σε {fb['reviewed']} ελέγχους από «τυφλό» χρήστη το Recall@2000 πήγε από {_ks(fb['without@2000'])} σε {_ks(fb['with@2000'])}, με {fb['test_clients_left']} κρυμμένους πελάτες: ανάγνωση ως «δεν βλάπτει», όχι ως κέρδος."})
    it = r.get("intent")
    if it:
        m, f = np.mean([x["model_p100"] for x in it]), np.mean([x["reasons_p100"] for x in it])
        out.append({"level": "good" if f > m else "info", "text": f"Με {it[0]['labels']} επισημάνσεις και λόγο απόρριψης, οι προτάσεις που ταιριάζουν στην πρόθεση του χρήστη πάνε κατά μέσο όρο από {100 * m:.0f}% σε {100 * f:.0f}% (προσομοίωση)."})
    return out


def run_suite(base, clients, quick=False, progress=lambda ev: None) -> dict:
    """progress receives a snapshot after every finished unit: done/total, elapsed, eta, the log and the results so far."""
    t0 = time.time()
    M = _Market(base, clients, B=30)
    if len(M.all_vats) < 30:
        return {"status": "error", "message": "Χρειάζονται τουλάχιστον 30 ταυτοποιημένοι ΑΦΜ για αυστηρό διαχωρισμό."}
    R.pu_features(base.model, base.companies["latitude"], base.companies["longitude"])     # once, before any thread starts
    split = make_split(M.all_vats, feedback=0.2, test=0.2)
    n_rep, intent_rounds, n_can, n_intent, blind_rounds = (3, 1, 3, 3, 1) if quick else (10, 2, 5, len(intent_concepts(base)), 3)
    plan = [] if quick else generalisation_plan(M)
    done, log, lock = [0], [], threading.Lock()
    total = 2 + n_rep + len(plan) + n_can + 1 + n_intent
    res = {"split": {"sizes": split.sizes(), "fingerprint": split.fingerprint(), "salt": split.salt}, "ks": list(C.KS), "universe": M.n, "quick": quick, "generalisation": []}

    def snapshot(step, message):
        el = time.time() - t0
        eta = None if done[0] == 0 else round(el / done[0] * (total - done[0]), 1)
        progress({"status": "running", "step": step, "message": message, "done": done[0], "total": total, "pct": round(100 * done[0] / total), "elapsed_s": round(el, 1),
                  "eta_s": eta, "log": log[-14:], "partial": dict(res, insights=insights(res))})

    def ticker(step):
        def tick(msg, weight=1):
            with lock:
                done[0] += weight; log.append(msg); snapshot(step, msg)
        return tick

    snapshot("locked", "Χτίζουμε τα προφίλ από το 80% των πελατών και ψάχνουμε το κρυμμένο 20%")
    tick = ticker("locked")
    res["locked"] = locked_test(M, split)
    tick(f"Κλειδωμένο τεστ: Recall@500 {100 * res['locked']['product@500']:.1f}%, Recall@2000 {100 * res['locked']['product@2000']:.1f}%", 2)
    snapshot("repeated", "Επαναλαμβάνουμε με άλλους διαχωρισμούς")
    res["repeated"] = repeated(M, n_rep, ticker("repeated"))
    if plan:
        snapshot("generalisation", "Κρύβουμε ολόκληρες περιοχές, κλάδους και προφίλ")
        res["generalisation"] = hard_generalisation(M, plan, ticker("generalisation"))
    snapshot("canaries", "Ελέγχουμε τον ίδιο τον έλεγχο για διαρροή")
    res["canaries"] = canaries(M, split, tick=ticker("canaries"), quick=quick)
    res["canaries_passed"] = all(c["passed"] for c in res["canaries"])
    snapshot("feedback", "Προσομοιώνουμε χρήστη που δεν γνωρίζει το κρυμμένο σύνολο")
    res["feedback_blind"] = [feedback_blind(M, split, rounds=blind_rounds)]
    ticker("feedback")("Τυφλός χρήστης: ολοκληρώθηκε")
    snapshot("intent", "Προσομοιώνουμε πωλητές με συγκεκριμένη πρόθεση")
    res["intent"] = intent_adaptation(M, split, rounds=intent_rounds, tick=ticker("intent"), generic=not quick, only=n_intent)
    res["insights"] = insights(res)
    return {"status": "done", **res, "log": log, "seconds": round(time.time() - t0, 1),
            "note": "Οι προηγούμενες εκδόσεις ρύθμισαν τα βάρη στους ίδιους πελάτες, οπότε ούτε το κλειδωμένο τεστ είναι εντελώς καθαρό για εκείνες τις επιλογές· "
                    "είναι καθαρό για κάθε νέα επιλογή (ανατροφοδότηση, έλεγχος χαρακτηριστικών), που έγινε μόνο στους υπόλοιπους πελάτες."}
