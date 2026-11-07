"""Glue: load data -> build blocks -> BlockModel. The universe part is built ONCE and cached by the API;
clients are attached per upload (so a new upload costs milliseconds, not minutes)."""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from . import features as F, ingest as I, embed as E, config as C, clarity as K, external as X


@dataclass
class Context:
    companies: pd.DataFrame
    activity: pd.DataFrame
    kad_desc: dict
    clients: pd.DataFrame | None
    client_shops: pd.DataFrame | None
    match_stats: dict | None
    primary_kad: np.ndarray
    model: F.BlockModel
    W: object
    kad_ids: list
    embed_backend: str = "tfidf"
    clarity: pd.DataFrame | None = None      # Activity Clarity Index per business (row order = companies)
    factors: pd.DataFrame | None = None      # external factors per business (by municipality), see external.py
    factor_catalog: list | None = None


def build_base(data_dir: Path, embed="tfidf", drop_other=None) -> Context:
    u = I.load_universe(data_dir)
    comp, act, desc = u["companies"], u["activity"], u["kad_desc"]
    kad_ids = sorted(set(act["activity_id"].dropna()))
    texts = [desc.get(k) or k for k in kad_ids]
    Emb = E.embed_texts(texts, embed)
    W = F.build_kad_weights(act, comp["company_id"], kad_ids, drop_other=drop_other, emb=Emb if C.NOISE_DAMP else None)
    blocks = {
        "activity": F.activity_block(W, Emb),
        "admin": F.admin_block(comp["prefecture_name"], comp["municipality_name"]),
        "geo": F.geo_block(comp["latitude"], comp["longitude"], comp["municipality_name"]),
        "legal": F.onehot_block(comp["legal_type"]),
        "age": F.age_block(comp["incorporation_date"]),
        "struct": F.struct_block(comp["is_branch"], comp["vat"]),
    }
    pk = F.primary_kad(act, comp["company_id"])
    clar = K.compute(act, comp["company_id"], kad_ids, Emb)
    fac, cat = X.build(comp, data_dir)
    return Context(comp, act, desc, None, None, None, pk, F.BlockModel(blocks), W, kad_ids, embed, clar, fac, cat)


def attach_clients(base: Context, clients: pd.DataFrame, match_mode="split") -> Context:
    shops, stats = I.match_clients(base.companies, clients, match_mode)
    return dataclasses.replace(base, clients=clients, client_shops=shops, match_stats=stats)


def build_context(data_dir: Path, clients_xlsx: Path | None = None, embed="tfidf", match_mode="split",
                  drop_other=None) -> Context:
    base = build_base(data_dir, embed, drop_other)
    xlsx = clients_xlsx or I.find_file(data_dir, "*Pelatologio*.xlsx")
    return attach_clients(base, I.load_clients(xlsx), match_mode)
