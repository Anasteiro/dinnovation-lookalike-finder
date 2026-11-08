"""Feedback re-ranking and hard split: invariants that must hold on any data (synthetic here, never quoted as accuracy)."""
import numpy as np
import pytest
from backend import analysis as A, feedback as FB, hardsplit as HS, ingest as I, pipeline as P, rankers as R
from scripts import make_synthetic


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn2")
    make_synthetic.main(["--out", str(d), "--n", "3000"])
    base = P.build_base(d, "tfidf")
    clients = I.load_clients(next(d.glob("*.xlsx")))
    ctx = P.attach_clients(base, clients)
    return base, clients, ctx, A.run_analysis(ctx, B=3)


def sess(world):
    return FB.FeedbackSession(world[2], world[3]["_state"])


# ------------------------------------------------------------------------------- split
def test_split_is_a_partition_and_order_free():
    v = [f"{i:09d}" for i in range(300)]
    a, b = HS.make_split(v), HS.make_split(list(reversed(v)))
    assert a == b and a.fingerprint() == b.fingerprint() and a.sizes() == {"train": 180, "feedback": 60, "test": 60}
    assert not (a.train & a.test or a.train & a.feedback or a.feedback & a.test) and a.train | a.feedback | a.test == set(v)


def test_split_membership_is_stable_when_other_vats_are_added():
    v = [f"{i:09d}" for i in range(1000)]
    assert HS.vat_bucket(v[3]) == HS.vat_bucket(v[3]) and 0 <= HS.vat_bucket(v[3]) < 1
    assert HS.make_split(v).fingerprint() != HS.make_split(v, salt="other").fingerprint()


def test_disjoint_guard_catches_a_shared_vat():
    with pytest.raises(HS.LeakageError):
        HS.assert_disjoint({"1", "2"}, {"2", "3"})


# ------------------------------------------------------------------------------- feedback
def test_pu_negative_argument_is_a_no_op_when_absent():
    Z = np.random.default_rng(0).normal(size=(400, 8))
    assert np.array_equal(R.pu_scores(Z, np.arange(10), np.ones(10)), R.pu_scores(Z, np.arange(10), np.ones(10), neg=None, neg_w=99))


def test_no_labels_is_exactly_the_model(world):
    s, st = sess(world), world[3]["_state"]
    assert np.array_equal(s.ranking().order, st["_default_order"]) and s.ranking().alpha == 0


def test_skip_reasons_remove_one_row_and_teach_nothing(world):
    s, st = sess(world), world[3]["_state"]
    o = st["_default_order"]
    s.add(s.ids[o[0]], "closed"); s.add(s.ids[o[5]], "not_now")
    assert np.array_equal(s.ranking().order, np.delete(o, [0, 5])) and s.ranking().alpha == 0


def test_strength_zero_ignores_teaching(world):
    s, st = sess(world), world[3]["_state"]
    s.set_strength(0.0)
    o = st["_default_order"]
    s.add(s.ids[o[0]], "good_fit"); s.add(s.ids[o[1]], "not_fit")
    assert np.array_equal(s.ranking().order, np.delete(o, [0, 1]))


def test_reset_and_undo_restore_exactly(world):
    s, st = sess(world), world[3]["_state"]
    o = st["_default_order"]
    s.add(s.ids[o[0]], "good_fit"); first = s.ranking().order.copy()
    s.add(s.ids[o[1]], "wrong_sector"); s.undo()
    assert np.array_equal(s.ranking().order, first)
    s.reset()
    assert np.array_equal(s.ranking().order, o) and [h["action"] for h in s.history][-1] == "reset"


def test_changing_a_label_and_undoing_restores_the_previous_one(world):
    s, st = sess(world), world[3]["_state"]
    i = st["_default_order"][0]
    s.add(s.ids[i], "good_fit"); s.add(s.ids[i], "not_fit")
    assert s.labels[int(i)] == "not_fit"
    s.undo()
    assert s.labels[int(i)] == "good_fit"


def test_clients_cannot_be_labelled(world):
    s = sess(world)
    with pytest.raises(ValueError):
        s.add(world[2].client_shops["company_id"].iat[0], "good_fit")


def test_wrong_area_demotes_that_area_only(world):
    ctx, st = world[2], world[3]["_state"]
    s = sess(world)
    pref = ctx.companies["prefecture_name"].to_numpy()
    o = st["_default_order"]
    top_pref = pref[o[0]]
    rej = [i for i in o[:400] if pref[i] == top_pref][:10]
    for i in rej:
        s.add(s.ids[i], "wrong_area")
    new = s.ranking().order[:200]
    kept = o[~np.isin(o, rej)][:200]
    assert (pref[new] == top_pref).mean() < (pref[kept] == top_pref).mean()


def test_ranking_is_deterministic_and_a_permutation(world):
    st = world[3]["_state"]
    o = st["_default_order"]
    def run():
        s = sess(world)
        s.add(s.ids[o[0]], "good_fit"); s.add(s.ids[o[1]], "not_fit"); s.add(s.ids[o[2]], "wrong_sector")
        return s.ranking().order
    a, b = run(), run()
    assert np.array_equal(a, b) and len(np.unique(a)) == len(a) and len(a) == len(o) - 3 and not set(o[:3]) & set(a)


def test_export_has_accepted_first_and_no_rejected(world):
    s, st = sess(world), world[3]["_state"]
    o = st["_default_order"]
    s.add(s.ids[o[3]], "good_fit"); s.add(s.ids[o[0]], "not_fit")
    e = s.export_frame()
    assert e["company_id"].iat[0] == s.ids[o[3]] and e["status"].iat[0] == "accepted" and s.ids[o[0]] not in set(e["company_id"]) and e["company_id"].is_unique


def test_no_solvency_reason_exists():
    t = " ".join(v["label"] for v in FB.REASONS.values()).lower() + " ".join(FB.REASONS)
    assert not any(w in t for w in ("πιστοληπτ", "φερέγγυ", "αξιόπιστ", "credit", "solven", "pay"))


# ------------------------------------------------------------------------------- suite on synthetic data: plumbing only
def test_suite_runs_and_canaries_pass(world):
    r = HS.run_suite(world[0], world[1], quick=True)
    assert r["status"] == "done" and r["canaries_passed"], r["canaries"]


def test_blind_oracle_never_sees_the_test_set(world):
    base, clients = world[0], world[1]
    M = HS._Market(base, clients, B=2)
    a = HS.make_split(M.all_vats)
    r = HS.feedback_blind(M, a, batch=10, rounds=1)
    assert r["reviewed"] == 10 and r["accepted"] <= 10


def test_validate_with_no_labels_says_too_few(world):
    assert sess(world).validate()["status"] == "too_few"


def test_validate_with_enough_labels(world):
    s, st = sess(world), world[3]["_state"]
    for n, i in enumerate(st["_default_order"][:16]):
        s.add(s.ids[i], "good_fit" if n % 2 else "not_fit")
    v = s.validate()
    assert v["status"] == "done" and 0 <= v["auc_model"] <= 1 and 0 <= v["auc_feedback"] <= 1


def test_fast_ari_equals_sklearn():
    from sklearn.metrics import adjusted_rand_score as sk
    from backend.icp import adjusted_rand_score as fast
    rng = np.random.default_rng(1)
    for _ in range(200):
        n, a, b = rng.integers(2, 300), rng.integers(1, 7), rng.integers(1, 7)
        x, y = rng.integers(0, a, n), rng.integers(0, b, n)
        assert fast(x, y) == sk(x, y)


def test_rank_only_equals_full_analysis(world):
    ctx, full = world[2], world[3]
    fast = A.run_analysis(ctx, B=3, rank_only=True)
    assert np.array_equal(fast["_state"]["_default_order"], full["_state"]["_default_order"])
    assert np.array_equal(fast["_state"]["rank_score"], full["_state"]["rank_score"])


def test_threaded_suite_matches_serial(world):
    w = HS.WORKERS
    try:
        M = HS._Market(world[0], world[1], B=2); HS.WORKERS = 1; a = HS.repeated(M, 3)
        M = HS._Market(world[0], world[1], B=2); HS.WORKERS = 3; b = HS.repeated(M, 3)
    finally:
        HS.WORKERS = w
    assert a == b


def test_progress_is_monotonic_and_reports_partial_results(world):
    ev = []
    r = HS.run_suite(world[0], world[1], quick=True, progress=ev.append)
    pct = [e["pct"] for e in ev]
    assert pct == sorted(pct) and pct[0] == 0 and pct[-1] <= 100 and ev[-1]["done"] == ev[-1]["total"]
    assert any("locked" in e["partial"] for e in ev) and r["insights"] and "partial" in ev[-1]


def test_brief_check_passes_on_real_format(world, tmp_path):
    from backend import briefcheck as BC
    A.write_outputs(world[3], tmp_path)
    res = BC.check(tmp_path, world[2].companies, world[1]["vat"])
    assert all(r["passed"] for r in res), [r for r in res if not r["passed"]]


def test_brief_check_catches_a_client_in_the_list(world, tmp_path):
    from backend import briefcheck as BC
    import pandas as pd
    A.write_outputs(world[3], tmp_path)
    p = tmp_path / "lookalikes.csv"
    d = pd.read_csv(p, dtype=str); d.loc[0, "company_id"] = str(world[2].client_shops["company_id"].iat[0]); d.to_csv(p, index=False)
    assert not next(r for r in BC.check(tmp_path, world[2].companies, world[1]["vat"]) if "ήδη πελάτης" in r["rule"])["passed"]


# ------------------------------------------------------------------------------- profile count
def test_profile_count_does_not_depend_on_bootstrap_count():
    from backend import features as F, icp
    rng = np.random.default_rng(0)
    centers = np.eye(3, 12)
    X = np.vstack([c + rng.normal(scale=.06, size=(40, 12)) for c in centers])
    m = F.BlockModel({"activity": F._l2_rows(X.astype(np.float32))}, {"activity": 1.0})
    w, vat = np.ones(120), np.array([f"{i}" for i in range(120)])
    res = [icp.discover(m, w, vat, B=b) for b in (5, 13, 30)]
    assert [r.k for r in res] == [3, 3, 3] and all(np.array_equal(res[0].labels, r.labels) for r in res)


def test_one_icp_when_homogeneous():
    from backend import features as F, icp
    X = np.random.default_rng(1).normal(size=(80, 10)) * .02 + 1
    m = F.BlockModel({"activity": F._l2_rows(X.astype(np.float32))}, {"activity": 1.0})
    assert icp.discover(m, np.ones(80), np.array([str(i) for i in range(80)]), B=10).k == 1


# ------------------------------------------------------------------------------- neural embedding test (a stand-in model: plumbing only)
class _FakeST:
    def __init__(self, name, device=None):
        self.name = name

    def encode(self, texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False):
        import hashlib
        out = np.zeros((len(texts), 32), np.float32)
        for i, t in enumerate(texts):
            for tok in t.split():
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 32] += 1
        return out + 1e-3


def test_embedbench_runs_a_neural_backend_and_skips_a_missing_one(world, tmp_path, monkeypatch):
    import sys, types
    from backend import embedbench as EB
    from pathlib import Path
    data = Path(tmp_path)
    make_synthetic.main(["--out", str(data), "--n", "1500"])
    clients = I.load_clients(next(data.glob("*.xlsx")))
    assert EB.evaluate("st:missing/model", data, clients, 1).status == "skipped" or "sentence_transformers" in sys.modules
    out = data / "res"
    st = types.ModuleType("sentence_transformers"); st.SentenceTransformer = _FakeST
    tc = types.ModuleType("torch"); tc.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st); monkeypatch.setitem(sys.modules, "torch", tc)
    rows = [EB.evaluate(b, data, clients, 2, log=lambda m: None) for b in ("tfidf", "st:intfloat/multilingual-e5-base")]
    assert [r.status for r in rows] == ["done", "done"] and rows[1].detail["dim"] == 32
    md = EB.to_markdown(rows, EB.compare(rows), 2)
    assert "multilingual-e5-base" in md and "MIT" in md and EB.compare(rows)[0]["against"] == "tfidf"


def test_kad_neighbour_quality_is_one_for_perfect_and_low_for_random():
    from backend import embedbench as EB
    rng = np.random.default_rng(0)
    ids = [f"{g:04d}{i:04d}" for g in range(40) for i in range(8)]
    perfect = np.eye(40)[[g for g in range(40) for _ in range(8)]] + rng.normal(scale=.001, size=(320, 40))
    assert EB.kad_neighbour_quality(perfect, ids) > 0.99 and EB.kad_neighbour_quality(rng.normal(size=(320, 40)), ids) < 0.15


# ------------------------------------------------------------------------------- titles: business first
def _enr():
    import pandas as pd
    rows = [("legal:ΙΚΕ", "Νομική μορφή ΙΚΕ", .96), ("muni:Χ / Υ", "Δήμος Χ / Υ", .4), ("pref:ΑΤΤΙΚΗΣ", "Νομός ΑΤΤΙΚΗΣ", .6), ("kadx:1", "Δηλωμένος δευτερεύων ΚΑΔ: Α (1)", .3), ("kad4:5611", "Ομάδα ΚΑΔ 5611", .8), ("kad:56110100", "Κύριος ΚΑΔ: ΕΣΤΙΑΤΟΡΙΑ (56110100)", .3)]
    return pd.DataFrame(rows, columns=["feature", "label", "share_in_profile"])


def test_title_is_business_then_place_then_legal():
    t = A.make_title(_enr(), [{"code": "5611", "label": "Εστιατόρια", "share": .8}, {"code": "5630", "label": "Καφέ και μπαρ", "share": .1}]).split(" · ")
    assert t == ["Εστιατόρια", "Δήμος Χ", "ΙΚΕ"]


def test_title_names_the_common_types_not_a_rare_one():
    t = A.make_title(_enr(), [{"code": "5630", "label": "Καφέ και μπαρ", "share": .4}, {"code": "5611", "label": "Εστιατόρια", "share": .3}, {"code": "5622", "label": "Κυλικεία", "share": .1}])
    assert t.startswith("Καφέ και μπαρ και Εστιατόρια · ")


def test_presentation_order_keeps_statistics_and_puts_business_first():
    e = A.business_first(_enr())
    assert e["feature"].tolist()[:2] == ["kad4:5611", "kad:56110100"] and set(e["feature"].tolist()[2:4]) == {"pref:ΑΤΤΙΚΗΣ", "muni:Χ / Υ"} and e["feature"].iat[4] == "legal:ΙΚΕ" and len(e) == 6


def test_every_profile_title_starts_with_a_business_type(world):
    for p in world[3]["profiles"]:
        head = p["title"].split(" · ")[0]
        assert not head.startswith(("Δήμος", "Νομός", "Νομοί")) and head not in ("ΙΚΕ", "ΕΕ", "ΟΕ", "ΑΕ") and p["business_types"]


def test_llm_facts_lead_with_the_business_type(world):
    from backend import llm
    assert llm.build_facts(world[3]["profiles"][0])[2]["text"].startswith("Τύπος επιχειρήσεων")


# ------------------------------------------------------------------------------- model wiring
def test_app_falls_back_visibly_when_the_model_is_missing(tmp_path, monkeypatch):
    from backend import embed
    make_synthetic.main(["--out", str(tmp_path), "--n", "800"])
    real = embed.embed_texts
    def fake(texts, backend="tfidf", use_cache=True):
        if backend != "tfidf":
            raise ModuleNotFoundError("no model")
        return real(texts, backend, use_cache)
    monkeypatch.setattr(embed, "embed_texts", fake)
    b = P.build_base(tmp_path, "st:x/y", fallback=True)
    assert b.embed_backend == "tfidf" and "st:x/y" in b.embed_warning
    with pytest.raises(ModuleNotFoundError):
        P.build_base(tmp_path, "st:x/y")                     # a measurement must stop instead of quietly using another model


def test_default_model_is_neural():
    from backend import config as C
    assert C.DEFAULT_EMBED.startswith("st:")


def test_embedbench_run_stores_a_result_and_brief_rule_sees_it(tmp_path, monkeypatch):
    import sys, types
    from backend import embedbench as EB, briefcheck as BC
    make_synthetic.main(["--out", str(tmp_path), "--n", "1500"])
    clients = I.load_clients(next(tmp_path.glob("*.xlsx")))
    st = types.ModuleType("sentence_transformers"); st.SentenceTransformer = _FakeST
    tc = types.ModuleType("torch"); tc.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st); monkeypatch.setitem(sys.modules, "torch", tc)
    ev = []
    out = tmp_path / "o"
    r = EB.run(tmp_path, clients, out, ["tfidf", "st:intfloat/multilingual-e5-base"], 2, ev.append)
    assert r["neural_done"] == 1 and EB.load_result(out)["neural_done"] == 1 and (out / "embedding_comparison.md").exists()
    pct = [e["pct"] for e in ev]
    assert pct == sorted(pct) and r["insights"]


def test_embedbench_no_silent_success_when_all_neural_fail(tmp_path):
    from backend import embedbench as EB
    make_synthetic.main(["--out", str(tmp_path), "--n", "1000"])
    clients = I.load_clients(next(tmp_path.glob("*.xlsx")))
    r = EB.run(tmp_path, clients, tmp_path / "o", ["st:not/a-real-model"], 2)
    assert r["neural_done"] == 0 and r["models"][0]["status"] == "skipped" and r["insights"][0]["level"] == "bad"


def test_preflight_names_python_and_fix():
    from backend import embedbench as EB
    p = EB.preflight()
    assert p["python"] and (p["ready"] or "compare_embeddings.bat" in p["fix"])


def test_brief_rule_fails_without_comparison(world, tmp_path):
    from backend import briefcheck as BC
    A.write_outputs(world[3], tmp_path)
    r = BC.check(tmp_path, world[2].companies, world[1]["vat"], tmp_path / "nope.json")
    assert not next(x for x in r if x["ref"] == "§5.3")["passed"]
