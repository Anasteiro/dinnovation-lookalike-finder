import json
import httpx
from backend import enrich as EN


def fake_places(request: httpx.Request) -> httpx.Response:
    assert request.headers["X-Goog-Api-Key"] == "SECRETKEY123" and "X-Goog-FieldMask" in request.headers
    body = json.loads(request.content)
    lat = body["locationBias"]["circle"]["center"]["latitude"]; lon = body["locationBias"]["circle"]["center"]["longitude"]
    if "NOMATCH" in body["textQuery"]:
        return httpx.Response(200, json={})
    far = {"id": "far", "displayName": {"text": "Far"}, "location": {"latitude": lat + 0.01, "longitude": lon}}
    near = {"id": "near", "displayName": {"text": "Near"}, "location": {"latitude": lat + 0.0002, "longitude": lon}, "rating": 4.5,
            "userRatingCount": 120, "websiteUri": "https://example.org", "googleMapsUri": "https://maps.google.com/?cid=1",
            "regularOpeningHours": {"openNow": True}}
    return httpx.Response(200, json={"places": [far, near]})


def test_enrich_matches_by_distance_and_resumes(tmp_path):
    rows = [{"company_id": "1", "name": "Taverna X", "municipality": "ΧΑΝΙΩΝ / ΧΑΝΙΩΝ", "lat": 35.5, "lon": 24.0},
            {"company_id": "2", "name": "Επιχείρηση Εστίασης", "municipality": "ΧΑΝΙΩΝ", "lat": 35.6, "lon": 24.1},
            {"company_id": "3", "name": "NOMATCH", "municipality": "Χ", "lat": 35.7, "lon": 24.2},
            {"company_id": "4", "name": "No gps", "municipality": "Χ", "lat": float("nan"), "lon": float("nan")}]
    f = tmp_path / "e" / "places.json"
    client = httpx.Client(transport=httpx.MockTransport(fake_places))
    db = EN.enrich(rows, "SECRETKEY123", f, client=client, pause=0)
    assert set(db["places"]) == {"1", "2"} and db["places"]["1"]["place_id"] == "near" and db["places"]["1"]["rating"] == 4.5
    assert db["places"]["2"]["distance_m"] <= 60 and db["stats"]["matched"] == 2          # near place 22 m away accepted, 1.1 km one rejected
    calls = []
    client2 = httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(1) or fake_places(r)))
    EN.enrich(rows, "SECRETKEY123", f, client=client2, pause=0)
    assert not calls                                                                         # resumable: nothing is fetched twice
    assert "SECRETKEY123" not in f.read_text(encoding="utf-8")                                          # the key is never stored


def test_generic_names_use_tight_radius():
    assert EN.query_for("Επιχείρηση Εστίασης", "Χ")[1] == 60.0
    assert EN.query_for("Taverna X", "ΧΑΝΙΩΝ / ΧΑΝΙΩΝ") == ("Taverna X ΧΑΝΙΩΝ", 150.0)
