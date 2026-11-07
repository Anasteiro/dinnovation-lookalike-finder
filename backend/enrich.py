"""Optional enrichment of the best lookalikes with Google Places data (rating, reviews, phone, website, opening status, link).

Rules we follow:
  * It runs ONCE before the demo (`python run.py enrich --key ...`), never while the app is running: the result is a local file,
    so the demo works without internet. The key is read from --key or the GOOGLE_PLACES_API_KEY environment variable and is never saved.
  * Google data is shown as extra information only, labelled with its source. It never feeds the scoring.
  * A place is accepted only if it is close to the business' own GPS point (so a wrong match is not shown as fact); the match rate is reported.
  * Google's terms limit how long data may be kept: the file stores the fetch date; the app shows it."""
from __future__ import annotations
import json
import math
import os
import time
from pathlib import Path

import httpx

URL = "https://places.googleapis.com/v1/places:searchText"
FIELDS = ",".join(["places.id", "places.displayName", "places.location", "places.rating", "places.userRatingCount",
                   "places.nationalPhoneNumber", "places.websiteUri", "places.googleMapsUri", "places.businessStatus",
                   "places.regularOpeningHours.openNow", "places.formattedAddress", "places.primaryTypeDisplayName"])
SOURCE = "Google Places API (New)"
GENERIC = ("επιχείρηση εστίασης", "επιχειρηση εστιασης")


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 12742000 * math.asin(math.sqrt(a))


def is_generic(name: str) -> bool:
    n = (name or "").strip().lower()
    return (not n) or n.startswith(GENERIC)


def query_for(name: str, municipality: str):
    """Named business -> search its name near its point; anonymised ('Επιχείρηση Εστίασης') -> search for restaurants in a tight radius."""
    if is_generic(name):
        return "restaurant", 60.0
    return f"{name} {str(municipality or '').split(' / ')[0]}".strip(), 150.0


def lookup(client: httpx.Client, key: str, name: str, municipality: str, lat: float, lon: float, lang="el") -> dict | None:
    q, radius = query_for(name, municipality)
    body = {"textQuery": q, "languageCode": lang, "maxResultCount": 5,
            "locationBias": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": radius}}}
    r = client.post(URL, json=body, headers={"X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELDS}, timeout=20)
    if r.status_code in (401, 403):
        raise PermissionError(f"Google refused the request ({r.status_code}): check the key, that 'Places API (New)' is enabled and billing is on.")
    r.raise_for_status()
    best, bd = None, 1e9
    for p in r.json().get("places", []):
        loc = p.get("location") or {}
        if "latitude" not in loc:
            continue
        d = haversine_m(lat, lon, loc["latitude"], loc["longitude"])
        if d < bd:
            best, bd = p, d
    if best is None or bd > radius:
        return None
    return {"place_id": best.get("id"), "name": (best.get("displayName") or {}).get("text"), "rating": best.get("rating"),
            "reviews": best.get("userRatingCount"), "phone": best.get("nationalPhoneNumber"), "website": best.get("websiteUri"),
            "maps_url": best.get("googleMapsUri"), "status": best.get("businessStatus"),
            "open_now": (best.get("regularOpeningHours") or {}).get("openNow"), "address": best.get("formattedAddress"),
            "type": (best.get("primaryTypeDisplayName") or {}).get("text"), "distance_m": round(bd), "named_query": not is_generic(name)}


def load(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {"source": SOURCE, "fetched": {}, "places": {}, "stats": {}}


def enrich(rows, key: str, path: Path, client: httpx.Client | None = None, pause: float = 0.05, log=print) -> dict:
    """rows: iterable of dicts (company_id, name, municipality, lat, lon). Resumable: companies already looked up are skipped."""
    path = Path(path)
    db = load(path)
    own = client is None
    client = client or httpx.Client()
    n = hit = 0
    try:
        for r in rows:
            cid = str(r["company_id"])
            if cid in db["fetched"]:
                continue
            if r.get("lat") is None or r.get("lon") is None or r["lat"] != r["lat"]:
                db["fetched"][cid] = None
                continue
            n += 1
            res = lookup(client, key, r["name"], r["municipality"], float(r["lat"]), float(r["lon"]))
            db["fetched"][cid] = time.strftime("%Y-%m-%d")
            if res:
                db["places"][cid] = res
                hit += 1
            if n % 25 == 0:
                log(f"  {n} looked up, {hit} matched")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")
            time.sleep(pause)
    finally:
        if own:
            client.close()
    tried = sum(1 for v in db["fetched"].values() if v)
    db["stats"] = {"looked_up": tried, "matched": len(db["places"]), "match_rate": round(len(db["places"]) / max(tried, 1), 3)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")
    return db


def main(argv=None):
    import argparse
    import pandas as pd
    from . import analysis as A, config as C
    from .pipeline import build_context
    ap = argparse.ArgumentParser(prog="run.py enrich", description="Fetch Google Places details for the best lookalikes (run once, before the demo).")
    ap.add_argument("--data", default=str(C.DATA_DIR))
    ap.add_argument("--embed", default="tfidf")
    ap.add_argument("--key", default=os.environ.get("GOOGLE_PLACES_API_KEY", ""), help="or set GOOGLE_PLACES_API_KEY (never stored)")
    ap.add_argument("--top", type=int, default=300, help="how many of the best global lookalikes (default 300)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if not a.key:
        raise SystemExit("No key: pass --key YOUR_KEY or set GOOGLE_PLACES_API_KEY.")
    data = Path(a.data)
    out = Path(a.out) if a.out else data / "enrichment" / "places.json"
    ctx = build_context(data, embed=a.embed)
    res = A.run_analysis(ctx, None)
    g = res["lookalikes_global"].head(a.top)
    comp = ctx.companies.set_index("company_id")
    rows = [{"company_id": c, "name": comp.at[c, "company_name_el"], "municipality": comp.at[c, "municipality_name"],
             "lat": comp.at[c, "latitude"], "lon": comp.at[c, "longitude"]} for c in g["company_id"]]
    print(f"Looking up {len(rows)} businesses (about {len(rows)} requests; cost depends on your Google plan). Output: {out}")
    db = enrich(rows, a.key, out)
    print("Done:", db["stats"])


if __name__ == "__main__":
    main()
