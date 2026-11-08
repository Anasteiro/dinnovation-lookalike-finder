"""Orchestrates the whole analysis: ICP discovery -> evidence -> assignments -> lookalikes -> files.
The same function will be called by the web API (Part 7)."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C, icp, stats, lookalike as L, rankers as R, ingest as I
from .labels import GROUP_NAMES, SHORT_LEGAL, short_kad
from . import clarity as K, external as X

CONF_NONE = 0.05          # below this share of own members -> "fits no profile"


def _clip(t: str, n: int) -> str:
    """Shortens a label at a word boundary without leaving an open bracket behind."""
    t = str(t)
    if len(t) <= n:
        return t
    t = t[:n].rsplit(" ", 1)[0].rstrip(" ,-(")
    if t.count("(") > t.count(")"):
        t = t[:t.rfind("(")].rstrip()
    return t + "…"


def _py(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    raise TypeError(type(o))


def short_place(label: str) -> str:
    """'Δήμος ΑΘΗΝΑΙΩΝ / ΚΕΝΤΡΙΚΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ' -> 'Δήμος ΑΘΗΝΑΙΩΝ'; 'Νομός ΑΤΤΙΚΗΣ' stays (official names, no guessing of accents)."""
    return label.split(" / ")[0]


def business_types(primary_kads, kad_desc: dict, top: int = 5) -> list[dict]:
    """What the clients of a profile ARE: share of each 4-digit main activity, most common first. Plain counts, no statistics."""
    s = pd.Series(primary_kads).astype(str)
    s = s[s != "NA"].str[:4]
    out = []
    for code, n in s.value_counts().head(top).items():
        d = kad_desc.get(code + "0000") or next((v for k, v in sorted(kad_desc.items()) if str(k).startswith(code)), code)
        out.append({"code": code, "label": GROUP_NAMES.get(code) or short_kad(d), "share": round(float(n / max(len(s), 1)), 3)})
    return out


FEATURE_GROUP = (("kad4:", "kad:"), ("pref:", "muni:"), ("legal:",))


def business_first(enr: pd.DataFrame) -> pd.DataFrame:
    """Presentation order: type of business, then where, then legal form, then the rest. Statistics are untouched; inside a group the
    most significant comes first. The first rows are balanced so that the description covers all three."""
    f = enr["feature"].astype(str)
    picks = []
    for prefixes, n in zip(FEATURE_GROUP, (3, 2, 1)):
        picks.extend(enr.index[f.str.startswith(prefixes)][:n].tolist())
    rest = [i for g in FEATURE_GROUP for i in enr.index[f.str.startswith(g)].tolist() if i not in picks]
    other = [i for i in enr.index if i not in picks and i not in rest]
    return enr.loc[picks + rest + other].reset_index(drop=True)


def make_title(enr: pd.DataFrame, types: list[dict] | None = None) -> str:
    """Business type · where · legal form, always in this order. The type is what most clients of the profile actually are."""
    if types:
        kept = [t for t in types if t["share"] >= 0.25][:2] or types[:1]
        names = list(dict.fromkeys(t["label"] for t in kept))
        kind = " και ".join(names)
    else:
        r = enr[enr["feature"].str.startswith(("kad:", "kad4:"))]
        kind = short_kad(r.iloc[0]["label"].split(": ", 1)[-1]) if len(r) else None
    def typical(prefix, min_share):
        r = enr[enr["feature"].str.startswith(prefix)]
        return r[r["share_in_profile"] >= min_share]
    muni, pref = typical("muni:", 0.3), enr[enr["feature"].str.startswith("pref:")].head(2)
    if len(muni):
        place = short_place(muni.iloc[0]["label"])
    elif len(pref):
        p = [x.replace("Νομός ", "") for x in pref["label"]]
        place = ("Νομός " if len(p) == 1 else "Νομοί ") + ", ".join(p)
    else:
        place = None
    leg = enr[enr["feature"].str.startswith("legal:")]
    keep = leg[leg["share_in_profile"] >= 0.3].head(2)
    keep = keep if len(keep) else leg.head(1)
    legal = ", ".join(SHORT_LEGAL.get(x.replace("Νομική μορφή ", "").strip(), x.replace("Νομική μορφή ", "").strip()) for x in keep["label"]) or None
    bits = [b for b in (kind, place, legal) if b]
    return " · ".join(bits) if bits else "Γενικό προφίλ πελατών"
    def first(prefix):
        r = enr[enr["feature"].str.startswith(prefix)]
        if prefix in ("muni:", "pref:") and len(r):
            r = r[r["share_in_profile"] >= 0.3]          # a place only names the profile if it is really typical of it
        return r.iloc[0]["label"] if len(r) else None
    kad = first("kad:")
    kad = short_kad(kad.split(": ", 1)[1]) if kad else None
    geo = first("muni:") or first("pref:")
    legal = first("legal:")
    if legal:
        legal = legal.replace("Νομική μορφή ", "").strip()
        legal = SHORT_LEGAL.get(legal, legal)
    bits = [b for b in (kad, legal, short_place(geo) if geo else None) if b]
    return " · ".join(bits) if bits else "Γενικό προφίλ πελατών"


def _times(x: float) -> str:
    return (f"{x:.1f}".replace(".", ",") if x < 10 else f"{x:.0f}") + " φορές"


def template_description(n, share, enr, types=None) -> str:
    """Automatic (non-AI) text built only from the statistics; q-values stay in the table below it."""
    txt = f"Ομάδα {n} πελατών" + (f" ({share:.0%} του τζίρου)." if share is not None else ".")
    if types:
        txt += " Τύπος επιχειρήσεων: " + ", ".join(f"{t['label']} ({t['share']:.0%})" for t in types[:3] if t["share"] >= 0.05) + "."
    if len(enr):
        top = "; ".join(f"{r.label} ({_times(r.shrunk_lift)} συχνότερα από την αγορά)" for r in enr.head(4).itertuples())
        txt += " Ξεχωρίζουν στατιστικά: " + top + "."
    else:
        txt += " Δεν εντοπίστηκε στατιστικά σημαντικό χαρακτηριστικό· το προφίλ βασίζεται στη συνολική ομοιότητα."
    return txt


def run_analysis(ctx, out_dir: Path | None = None, k=None, B=30, revenue_mode="log", top_per_profile=2000,
                 top_global=5000, blend=C.CENTROID_BLEND, m_knn=C.KNN_M, progress=lambda step, **kw: None, rank_only: bool = False) -> dict:
    """rank_only=True: same clustering and same ranking, without statistics, reasons and files (used by the hold-out tests)."""
    t0 = time.time()
    comp = ctx.companies
    n = len(comp)
    s = ctx.client_shops
    cpos = s["pos"].to_numpy()
    w = R.revenue_weights(s["weight"].to_numpy(float), revenue_mode)   # per-shop weight: a VAT's turnover is split across its shops (identical when each VAT has one shop)
    timings = {}

    progress("clustering")
    t = time.time()
    sub = ctx.model.subset(cpos)
    res = icp.discover(sub, w, s["vat"].to_numpy(), B=B, force_k=k)
    kk, labels = res.k, res.labels
    timings["clustering_s"] = round(time.time() - t, 2)

    progress("statistics")
    t = time.time()
    # market-only work cached on the model (never depends on the client list): feature matrix + VAT lookup
    mk = getattr(ctx.model, "_market_cache", None)
    if mk is None or mk["n"] != n:
        codes, uniq = pd.factorize(comp["vat"])
        mk = {"n": n, "fm": None, "vat_codes": codes, "vat_index": pd.Index(uniq)}
        ctx.model._market_cache = mk
    if not rank_only and mk["fm"] is None:
        mk["fm"] = stats.build_feature_matrix(comp, ctx.activity, ctx.kad_desc)
    F, names, flabels = (None, None, None) if rank_only else mk["fm"]
    cl_codes = mk["vat_index"].get_indexer(pd.Index(s["vat"].dropna().unique()))
    is_client = np.isin(mk["vat_codes"], cl_codes[cl_codes >= 0])
    bg_pos = np.flatnonzero(~is_client)
    timings["features_s"] = round(time.time() - t, 2)

    progress("scoring")
    t = time.time()
    score = np.zeros((n, kk), dtype=np.float32)
    near = np.zeros((n, kk), dtype=np.int64)
    rank_score = np.zeros((n, kk), dtype=np.float64)      # ranking score = 0.2*kNN + 0.8*PU-learning (both as percentiles)
    loo_sorted, enrs, members = {}, {}, {}
    Zpu = R.pu_features(ctx.model, comp["latitude"], comp["longitude"])
    for c in range(kk):
        mem = np.flatnonzero(labels == c)
        members[c] = mem
        mpos, mw = cpos[mem], w[mem]
        sc, nr = L.profile_scores(ctx.model, mpos, mw, m_knn, blend)
        score[:, c], near[:, c] = sc, mpos[nr]
        rank_score[:, c] = C.RANK_W_KNN * R._pct(sc) + C.RANK_W_PU * R._pct(R.pu_scores(Zpu, mpos, mw / mw.mean()))
        if not rank_only:
            loo_sorted[c] = np.sort(L.member_loo_scores(sub.subset(mem), mw, m_knn, blend))
            enrs[c] = business_first(stats.enrichment(F, names, flabels, mpos, bg_pos))
    timings["scoring_s"] = round(time.time() - t, 2)

    if rank_only:
        cand_ = np.flatnonzero(~is_client)
        sw = np.array([w[members[c]].sum() for c in range(kk)], float)
        g_ = R.merge_proportional(rank_score[cand_], sw / sw.sum())
        return {"meta": {"k": kk}, "_state": {"members": [(cpos[members[c]], w[members[c]]) for c in range(kk)], "share": sw / sw.sum(), "cand": cand_,
                                              "_default_order": cand_[np.argsort(-g_, kind="stable")], "_default_score": np.sort(g_)[::-1],
                                              "rank_score": rank_score, "profile_ids": [f"icp-{c + 1}" for c in range(kk)]}}

    # ---- assignments (A3): same score for everybody, confidence = share of own members scoring lower
    best = score.argmax(axis=1)
    best_score = score[np.arange(n), best]
    conf = np.zeros(n)
    for c in range(kk):
        m_ = best == c
        conf[m_] = L.confidence_from(loo_sorted[c], best_score[m_])
    pid = np.array([f"icp-{c + 1}" for c in range(kk)])
    status = np.where(conf < CONF_NONE, "none", "assigned")
    assign = pd.DataFrame({"company_id": comp["company_id"], "profile_id": np.where(status == "none", "none", pid[best]),
                           "score": best_score.round(5)})
    detail = assign.assign(vat=comp["vat"].values, confidence=conf.round(4), status=status, is_client=is_client)
    # agreement between clustering label and scoring label on the clients themselves
    agree = float((best[cpos] == labels).mean())

    # ---- lookalikes (A7)
    progress("lookalikes")
    cand = np.flatnonzero(~is_client)
    names_arr = comp["company_name_el"].fillna("").to_numpy()
    rows = []
    kad_lab = {k: (ctx.kad_desc.get(k) or k) for k in set(ctx.primary_kad)}
    mun_arr = comp["municipality_name"].fillna("").to_numpy()
    leg_arr = comp["legal_type"].fillna("").astype(str).to_numpy()

    clar_lvl = ctx.clarity["level"].to_numpy() if ctx.clarity is not None else None
    clar_val = ctx.clarity["clarity"].to_numpy() if ctx.clarity is not None else None
    clar_unrel = ctx.clarity["n_unrelated"].to_numpy() if ctx.clarity is not None else None

    def own_desc(i):                      # what is specific to THIS business (makes every row different)
        kd = str(ctx.primary_kad[i])
        parts = [f"κύριος ΚΑΔ: {_clip(kad_lab.get(kd, kd), 60)}"] if kd != "NA" else []
        if mun_arr[i]:
            parts.append(str(mun_arr[i]).split(" / ")[0])
        if leg_arr[i] and leg_arr[i] != "nan":
            parts.append(leg_arr[i])
        if clar_lvl is not None and clar_lvl[i] == "Θορυβώδες":
            parts.append(f"προσοχή: θορυβώδεις δηλωμένοι ΚΑΔ ({clar_unrel[i]} άσχετοι, καθαρότητα {clar_val[i]}/100)")
        return "Η επιχείρηση: " + ", ".join(parts) if parts else None
    # ---- global ranking (one list, each business once): proportional-budget merge of the per-profile rankings
    share_w = np.array([w[members[c]].sum() for c in range(kk)], float)
    gscore = R.merge_proportional(rank_score[cand], share_w / share_w.sum())
    shr = np.maximum(share_w / share_w.sum(), 1e-9)
    prank = np.stack([np.argsort(np.argsort(-rank_score[cand, c], kind="stable"), kind="stable") for c in range(kk)], axis=1)
    best_rank = prank.min(axis=1) + 1                       # best 1-based position of each business in ANY profile's own ranking
    gbest = np.argmin(prank / shr[None, :], axis=1)         # the profile that actually selected it in the merge (same key as merge_proportional)
    go = np.argsort(-gscore, kind="stable")[:top_global]

    # ---- reasons: for every per-profile top row AND every global row (with the profile that selected it)
    reason = {}
    per_profile_order = {}
    for c in range(kk):
        order = cand[np.argsort(-rank_score[cand, c], kind="stable")][:top_per_profile]
        per_profile_order[c] = order
        need = np.union1d(order, cand[go][gbest[go] == c])
        enr = enrs[c]
        top_f = enr.head(6)
        f_label = top_f["label"].astype(str).tolist()                 # plain lists: .iloc inside the loop was slow
        f_lift = top_f["shrunk_lift"].astype(float).tolist()
        fidx = pd.Index(names).get_indexer(top_f["feature"]) if len(top_f) else np.array([], int)
        Fc = F[need][:, fidx].toarray() if len(fidx) else np.zeros((len(need), 0))
        for r_, i in enumerate(need):
            on = np.flatnonzero(Fc[r_])
            on_set = set(on.tolist())
            off = [j for j in range(Fc.shape[1]) if j not in on_set]
            hit = [f"{f_label[j]} (×{f_lift[j]:.1f})" for j in on[:3]]
            miss = [f_label[j] for j in off[:2]]
            reason[(c, int(i))] = L.reason_text(hit, float(score[i, c]), names_arr[near[i, c]] or None, miss, Fc.shape[1], own_desc(i))
    for c in range(kk):
        order = per_profile_order[c]
        for r, (i, sc_) in enumerate(zip(order, rank_score[order, c]), start=1):
            rows.append((pid[c], comp["company_id"].iat[i], r, round(float(sc_), 5), reason[(c, int(i))]))
    look = pd.DataFrame(rows, columns=["profile_id", "company_id", "rank", "score", "αιτιολόγηση"])
    gidx = cand[go]
    glook = pd.DataFrame({"rank": np.arange(1, len(go) + 1), "company_id": comp["company_id"].to_numpy()[gidx],
                          "profile_id": pid[gbest[go]], "score": (1 - np.arange(len(go)) / max(len(cand), 1)).round(5),
                          "name": names_arr[gidx], "municipality": comp["municipality_name"].to_numpy()[gidx],
                          "vat": comp["vat"].to_numpy()[gidx], "profile_rank": best_rank[go],
                          "αιτιολόγηση": [reason[(int(c), int(i))] for c, i in zip(gbest[go], gidx)]})

    # ---- profiles.json
    real_w = s["weight"].to_numpy(float)
    tot_w = np.nansum(real_w)
    profiles = []
    for c in range(kk):
        mem = members[c]
        share = float(np.nansum(real_w[mem]) / tot_w) if tot_w > 0 and s["revenue"].notna().any() else None
        stab = float(res.stability.get(c, 1.0))
        enr = enrs[c]
        types = business_types(ctx.primary_kad[cpos[mem]], ctx.kad_desc)
        profiles.append({
            "profile_id": pid[c], "title": make_title(enr, types), "business_types": types,
            "description": template_description(int(len(mem)), share, enr, types), "description_is_ai": False,
            "description_meta": {"source": "template", "verified": False, "model": None, "paragraphs": [],
                                 "ai_label": "Αυτόματο κείμενο από στατιστικά (χωρίς AI)"},
            "n_clients": int(len(mem)), "n_client_vats": int(s["vat"].iloc[mem].nunique()),
            "turnover_share": share, "stability": round(stab, 3),
            "confidence": round(stab * min(1.0, len(mem) / 30), 3),
            "activity_clarity": K.profile_summary(ctx.clarity, cpos[mem]) if ctx.clarity is not None else None,
            "external_factors": X.profile_summary(ctx.factors, ctx.factor_catalog, cpos[mem]) if getattr(ctx, "factors", None) is not None else [],
            "features": [{"feature": r.feature, "label": r.label, "support": int(r.support),
                          "share_in_profile": float(r.share_in_profile), "share_in_universe": float(r.share_in_universe),
                          "lift": float(r.lift), "shrunk_lift": float(r.shrunk_lift),
                          "p_value": float(r.p_value), "q_value": float(r.q_value)} for r in enr.head(15).itertuples()],
        })
    meta = {"k": kk, "selection_rule": res.rule, "k_table": res.k_table, "match": ctx.match_stats,
            "embedding": ctx.embed_backend, "universe": int(n), "candidates": int(len(cand)), "assigned_to_profile": int((status == "assigned").sum()),
            "none_share": float((status == "none").mean()), "cluster_vs_score_agreement_on_clients": agree,
            "params": {"revenue_mode": revenue_mode, "bootstrap": B, "blend": blend, "m_knn": m_knn,
                       "conf_none": CONF_NONE, "feature_note": stats.MONTHS_NOTE},
            "timings": {**timings, "total_s": round(time.time() - t0, 2)}}
    out = {"profiles": profiles, "assignments": assign, "assignments_detail": detail, "lookalikes": look,
           "lookalikes_global": glook, "meta": meta,
           "activity_clarity": (ctx.clarity.reset_index().assign(explanation=lambda d: K.explain_all(d))
                                .rename(columns={"n_other": "declared_other_kads", "n_unrelated": "unrelated_kads"})
                                [["company_id", "clarity", "level", "declared_other_kads", "unrelated_kads", "explanation"]]
                                if ctx.clarity is not None else None),
           "_state": {"members": [(cpos[members[c]], w[members[c]]) for c in range(kk)], "share": share_w / share_w.sum(), "cand": cand, "_default_order": cand[np.argsort(-gscore, kind="stable")],
                      "_default_score": np.sort(gscore)[::-1], "rank_score": rank_score, "profile_ids": list(pid)}}
    if out_dir is not None:
        write_outputs(out, Path(out_dir))
    return out


def write_outputs(out: dict, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / "profiles.json").write_text(json.dumps(out["profiles"], ensure_ascii=False, indent=2, default=_py), encoding="utf-8")
    (d / "analysis_meta.json").write_text(json.dumps(out["meta"], ensure_ascii=False, indent=2, default=_py), encoding="utf-8")
    out["assignments"].to_csv(d / "assignments.csv", index=False, encoding="utf-8")
    out["assignments_detail"].to_csv(d / "assignments_detail.csv", index=False, encoding="utf-8")
    # lookalikes.csv = THE deliverable: ONE list, each business once. File order and `score` (descending) give the single
    # global ranking; `rank` is the business's best position in any profile's own ranking and `profile_id` that profile.
    # So "first k rows", "sorted by score" and "rank <= k" (per-profile reading, de-duplicated) are all meaningful.
    g = out["lookalikes_global"]
    # deliverable: ONE global ranking. `rank` is the global position 1..N (unique) and `score` falls with rank, so
    # "rank <= k" and "first k rows" select exactly the same k businesses (Recall@k is unambiguous).
    g[["profile_id", "company_id", "rank", "score", "αιτιολόγηση"]].to_csv(d / "lookalikes.csv", index=False, encoding="utf-8")
    out["lookalikes"].to_csv(d / "lookalikes_by_profile.csv", index=False, encoding="utf-8")      # per-profile lists (2000 each)
    out["lookalikes_global"].to_csv(d / "lookalikes_global.csv", index=False, encoding="utf-8")
    if out.get("activity_clarity") is not None:
        out["activity_clarity"].to_csv(d / "activity_clarity.csv", index=False, encoding="utf-8")
