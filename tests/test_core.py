import numpy as np, pandas as pd, pytest
from pathlib import Path
from backend import ingest as I, evalharness as H, features as F
from scripts import make_synthetic

@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    make_synthetic.main(["--out", str(d), "--n", "3000"])
    return d

def test_vat_normalisation():
    assert I.normalize_vat("023456789") == "023456789"
    assert I.normalize_vat("23456789") == "023456789"
    assert I.normalize_vat("EL 023-456-789") == "023456789"
    assert I.normalize_vat(None) is None

def test_load_and_match(synth):
    u = I.load_universe(synth)
    assert u["companies"]["latitude"].notna().mean() > 0.9
    cl = I.load_clients(next(synth.glob("*.xlsx")))
    shops, st = I.match_clients(u["companies"], cl)
    assert st["unmatched_vats"] == 0 and st["client_shops"] >= st["matched_vats"]
    # split mode never double counts turnover
    assert shops.groupby("vat")["weight"].sum().sub(cl.set_index("vat")["revenue"]).abs().max() < 1e-6

def test_kad_weights_fixed_mass():
    act = pd.DataFrame({"company_id": ["a"] * 6 + ["b"], "atype": ["primary"] + ["secondary"] * 5 + ["primary"],
                        "activity_id": ["1", "2", "3", "4", "5", "6", "1"]})
    W = F.build_kad_weights(act, pd.Series(["a", "b"]), list("123456"))
    assert np.allclose(W.sum(axis=1).A1, [1.5, 1.0])      # many declared KADs are not inflated

def test_perfect_scorer_has_full_recall():
    n = 1000; pos = np.arange(0, 100); w = np.ones(100); g = pos.astype(str)
    truth = np.zeros(n); truth[pos] = 1.0
    df = H.cross_validate(n, pos, w, g, {"oracle": lambda tr, w_: truth}, repeats=3, ks=(50,))
    assert (df["recall@50"] == 1.0).all()


def test_benjamini_hochberg_monotone():
    from backend.stats import benjamini_hochberg
    p = np.array([0.001, 0.04, 0.03, 0.5])
    q = benjamini_hochberg(p)
    assert (q >= p).all() and q[0] <= q[2] <= q[1] + 1e-12 and q.max() <= 1.0


def test_enrichment_finds_planted_feature_and_not_rare_noise():
    import scipy.sparse as sp
    from backend.stats import enrichment
    rng = np.random.default_rng(0)
    n = 5000
    planted = rng.random(n) < 0.1
    mem = np.arange(100); planted[mem] = rng.random(100) < 0.7        # strongly enriched among members
    rare = np.zeros(n, bool); rare[mem[:3]] = True                    # appears only 3 times, all in members
    noise = rng.random(n) < 0.2
    F = sp.csr_matrix(np.stack([planted, rare, noise], 1).astype(np.float32))
    names = np.array(["kad:A", "kad:rare", "kad:noise"])
    lab = {k: k for k in names}
    out = enrichment(F, names, lab, mem, np.arange(100, n), min_support=4)
    assert list(out["feature"]) == ["kad:A"]                          # rare below support, noise not significant


def test_kernel_kmeans_separates_two_blocks():
    from backend.icp import kernel_kmeans
    K = np.full((20, 20), 0.1); K[:10, :10] = 0.9; K[10:, 10:] = 0.9; np.fill_diagonal(K, 1.0)
    lab = kernel_kmeans(K, np.ones(20), 2, np.random.default_rng(1))
    assert len(set(lab[:10])) == 1 and len(set(lab[10:])) == 1 and lab[0] != lab[10]


def test_full_analysis_outputs(synth, tmp_path):
    from backend.pipeline import build_context
    from backend.analysis import run_analysis
    ctx = build_context(synth, embed="tfidf")
    out = run_analysis(ctx, tmp_path, B=5, top_per_profile=300)
    a = pd.read_csv(tmp_path / "assignments.csv"); l = pd.read_csv(tmp_path / "lookalikes.csv")
    assert list(a.columns) == ["company_id", "profile_id", "score"] and len(a) == len(ctx.companies)
    assert list(l.columns) == ["profile_id", "company_id", "rank", "score", "αιτιολόγηση"]
    clients = set(ctx.client_shops["company_id"])
    assert not (set(l["company_id"]) & clients)
    # one global ranking: each business once, rank 1..N in file order, score never increasing
    assert l["company_id"].is_unique and (l["rank"] >= 1).all()
    assert (l["score"].diff().dropna() <= 0).all() and l["αιτιολόγηση"].notna().all()
    bp0 = pd.read_csv(tmp_path / "lookalikes_by_profile.csv")
    assert (l["rank"].to_numpy() == range(1, len(l) + 1)).all()      # `rank` = unique global position: "rank <= k" == first k rows
    bp = pd.read_csv(tmp_path / "lookalikes_by_profile.csv")
    assert set(bp["profile_id"]) == {p["profile_id"] for p in out["profiles"]}
    assert all(0 <= p["stability"] <= 1 for p in out["profiles"])


# ------------------------------------------------------------------ LLM layer (mocked model)
def _profile():
    return {"profile_id": "icp-1", "n_clients": 124, "stability": 0.88, "turnover_share": 0.36,
            "features": [{"label": "Κύριος ΚΑΔ: Ταβέρνες (56101200)", "share_in_profile": 0.55, "share_in_universe": 0.12,
                          "shrunk_lift": 4.6}]}


def _sec(name):
    from backend import llm
    f = llm.build_facts(_profile())
    return f, {s["key"]: s for s in llm.build_sections(f)}[name]


def test_verifier_accepts_grounded_section():
    from backend import llm
    f, s = _sec("who")
    good = "Το προφίλ αποτελείται από 124 πελάτες που φέρνουν το 36% του τζίρου. Οι ταβέρνες είναι το 55% έναντι 12% της αγοράς."
    assert llm.verify_section(good, f, s) == []


def test_verifier_rejects_invented_things():
    from backend import llm
    f, s = _sec("who")
    assert any("αριθμός" in v for v in llm.verify_section("Έχει 300 πελάτες.", f, s))
    assert any("απαγορευμένος" in v for v in llm.verify_section("Είναι αξιόπιστοι πελάτες με 124 μέλη.", f, s))
    assert any("εκτός στοιχείων" in v for v in llm.verify_section("Είναι ιδιοκτήτες στην Πάτρα και στη Θεσσαλονίκη.", f, s))
    assert any("ελληνικό" in v for v in llm.verify_section("This profile has 124 clients today.", f, s))
    assert any("επανάληψη" in v for v in llm.verify_section("Έχει 124 πελάτες. Έχει 124 πελάτες.", f, s))
    f, a = _sec("approach")
    assert any("αριθμοί" in v for v in llm.verify_section("Ο πωλητής μπορεί να καλέσει 124 ταβέρνες.", f, a))
    f, st = _sec("stands_out")
    assert any("σταθερότητα" in v for v in llm.verify_section("Οι πελάτες είναι σταθεροί, 88%.", f, st))


def test_narrate_repairs_per_section_and_falls_back(tmp_path):
    from backend import llm
    calls = []
    def gen(prompt, system=llm.SYSTEM):
        calls.append(prompt)
        if "σταθερά ξαναβγαίνει" in prompt:                      # 2nd paragraph: always invalid (invented number)
            return "Η ομάδα ξαναβγαίνει στο 77% των φορών.", {"seconds": 0}
        if len(calls) == 1:                                      # 1st paragraph: bad, then good
            return "Έχει 999 πελάτες.", {"seconds": 0}
        return "Το προφίλ αποτελείται από 124 πελάτες.", {"seconds": 0}
    gen.model = "mock"
    r = llm.narrate(_profile(), gen, tmp_path)
    assert r["is_ai"] and not r["verified"] and r["source"] == "mixed"        # paragraph 1 AI, 2 fallback
    ps = r["paragraphs"]
    assert ps[0]["is_ai"] and ps[0]["attempts"] == 2 and not ps[1]["is_ai"]
    assert ps[2]["deterministic"] and not ps[2]["is_ai"] and "Προτεινόμενη στόχευση" in ps[2]["text"] and "Ταβέρνες" in ps[2]["text"]
    assert "999" not in r["text_clean"] and "77%" not in r["text_clean"] and "χωρίς AI" in r["ai_label"]
    assert llm.narrate(_profile(), gen, tmp_path)["cached"]                   # reproducible + cached
    always_bad = lambda prompt, system=llm.SYSTEM: ("Έχει 999 πελάτες.", {})
    r2 = llm.narrate(_profile(), always_bad)
    assert r2["source"] == "template" and r2["is_ai"] is False and r2["ai_label"].endswith("(χωρίς AI)")
    good = lambda prompt, system=llm.SYSTEM: ("Το προφίλ αποτελείται από 124 πελάτες.", {"seconds": 0}) if "ποιοι είναι" in prompt else \
        ("Η ομάδα ξαναβγαίνει σταθερά στο 88% των επαναδειγματοληψιών.", {"seconds": 0})
    r3 = llm.narrate(_profile(), good)
    assert r3["verified"] and r3["is_ai"] and "παράγραφοι 1, 2" in r3["ai_label"]


def test_pu_and_proportional_merge():
    from backend import rankers as R
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 8)); Z[:20, 0] += 3          # the 20 clients differ along feature 0
    sc = R.pu_scores(Z, np.arange(20), np.ones(20))
    assert sc[:20].mean() > sc[20:].mean() + 1
    # a profile with 3x the weight gets about 3x the slots in the global top-k
    a = np.stack([np.linspace(1, 0, 400), np.linspace(1, 0, 400)[::-1]], axis=1)
    g = R.merge_proportional(a, np.array([0.75, 0.25]))
    top = np.argsort(-g)[:100]
    assert (top < 200).sum() > 2 * (top >= 200).sum()


def test_clarity_index():
    import pandas as pd
    from backend import clarity as K
    act = pd.DataFrame({"company_id": ["a", "b", "b", "b", "c", "c", "c", "c", "c", "c"],
                        "activity_id": ["5610", "5610", "5630", "5621", "5610", "1011", "4711", "7311", "8510", "4941"],
                        "atype": ["primary", "primary", "secondary", "secondary", "primary"] + ["secondary"] * 5})
    kads = ["1011", "4711", "4941", "5610", "5621", "5630", "7311", "8510"]
    rng = np.random.default_rng(0); emb = rng.normal(size=(len(kads), 16))
    out = K.compute(act, pd.Series(["a", "b", "c", "d"]), kads, emb)
    assert out.loc["a", "clarity"] == 100 and out.loc["b", "clarity"] == 100          # same division codes are related
    assert out.loc["c", "clarity"] < out.loc["b", "clarity"] and out.loc["c", "n_other"] == 5
    assert out.loc["d", "clarity"] == 85                                                # no main code declared
