# Local ICP engine API (v0.4) - contract for the front-end

Base URL: `http://localhost:8000` (configurable). CORS is open, including Chrome's private-network preflight.
Everything runs locally. No endpoint calls an external AI service. CORS accepts only pages served from localhost/127.0.0.1 (set `FORCECRM_ALLOW_ORIGINS` to override).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `data_found`, `market_data_loaded`, `ollama_running`, `llm_model`, `embedding` |
| POST | `/api/analyze` | multipart: `file` (.xlsx/.csv, columns ΑΦΜ, Τζίρος) **or** `use_sample=true`; optional `k` (force #profiles), `boot` (default 30). Returns `{job_id}`. 400 with a readable message on bad files. |
| GET | `/api/jobs/{id}` | `{status: queued|running|done|error, error, events[], narration}` (polling) |
| GET | `/api/jobs/{id}/events` | Server-Sent Events. Each `data:` is `{t, step, ...}`; steps: uploaded, loading_market_data, matching_clients, matched, clustering, statistics, scoring, lookalikes, done / error, narrated, narration_done. Ends with `event: end` as soon as the ANALYSIS is done (narration is NOT streamed: poll `GET /api/jobs/{id}` -> `narration`). |
| GET | `/api/results/{id}` | `{dataset, meta, profiles[]}` (409 while running, 422 on error) |
| GET | `/api/lookalikes?job=&profile=all|icp-N&limit=&offset=&q=` | `{total, items[]}`; item: rank, company_id, profile_id, score, reason, name, municipality, prefecture, legal_type, primary_kad, primary_kad_label, founded, lat, lon |
| GET | `/api/geo/points?job=&kind=clients|lookalikes&profile=&limit=` | map points: company_id, profile_id, name, municipality, lat, lon |
| POST | `/api/jobs/{id}/narrate` | form `model` (optional if server started with `--llm-model`). Starts Greek AI descriptions in the background (this also happens AUTOMATICALLY after every analysis when the server runs with `--llm-model` and Ollama is up; `narration.status`: idle|running|done|error with `done`/`total`); progress comes as `narrated` events and in `GET /api/jobs/{id}` -> `narration`. |
| POST | `/api/jobs/{id}/benchmark` | form `repeats` (2-30, default 8). Starts the live accuracy check in the background (about 10 s): hides 20% of the uploaded clients per split, rebuilds with the product's method, measures Recall@500/2000 vs baseline and kNN. |
| GET | `/api/jobs/{id}/benchmark` | `{status: idle|running|done|error}`; when done: `methods{baseline,knn,product}{label, recall@500, recall@500_ci95, recall@2000, recall@2000_ci95, median_rank_pct}`, `lift_vs_baseline{500,2000}`, `paired_wins` (share of splits won, number 0-1), `paired_wins_by_k`, `random_level`, `repeats`, `n_client_vats`, `n_hidden_per_split`, `universe`, `seconds`, `note` (Greek caveat). Also included in `GET /api/jobs/{id}` as `benchmark`. |
| GET | `/api/runs` | saved runs from the SQLite store (`outputs/jobs/runs.db`): run_id, created_at, source, k, n_clients, matched_vats, universe |
| GET | `/api/geo/density?job=&kind=market|clients|lookalikes&profile=all|icp-N&cell=0.1&limit=2000` | Offline map data: `{cell_deg, total, bounds{lat[2],lon[2]}, cells[{lat, lon, n}]}`; cell size in degrees (0.01-1). No map tiles are used. |
| GET | `/api/factors?job=` | filter page catalogue: `{factors[{key,label,unit,source,higher,coverage,min,max,market_median}], options{legal[{value,count}],prefectures[],founded[2],branches}, profiles[{profile_id,title,external_factors[{key,label,profile_median,market_median,p_value,significant}]}], note}` |
| POST | `/api/jobs/{id}/refine` | JSON `{profile, filters{legal[],prefectures[],min_clarity,founded_from,founded_to,exclude_branches,factors{key:{min,max}}}, boosts{key:-1..1}, strength 0..1, limit, offset}` -> `{total, removed{label:n}, items[...model_rank, refined_score, factor columns], top2000_medians, export}`; writes `refined_lookalikes.csv` (export). No filters = the model ranking. |
| GET | `/api/clarity?job=` | Activity Clarity Index overview: `{method, levels{Καθαρό,Μέτριο,Θορυβώδες}, histogram{edges,counts}, market_mean, profiles{id: summary}, note}` |
| GET | `/api/clarity?job=&company_id=` | one business: `{clarity 0-100, level, declared_other_kads, unrelated_kads, explanation, kads[{code,type,description,related}], kads_total}`. Lookalike items also carry `clarity`, `clarity_level`, `declared_other_kads`, `unrelated_kads`; export `activity_clarity.csv`. |
| GET | `/api/geo/outline` | Land outline of Greece as a GeoJSON Feature (MultiPolygon, coordinates [lon, lat]), bundled with the app (Natural Earth, public domain). For drawing a Google-Maps-like base map offline. |
| GET | `/api/geo/municipalities?job=&profile=all|icp-N&limit=12&top=2000` | municipalities ranked by number of top-`top` lookalikes: `{items[{municipality, lookalikes, market, share_of_market_in_top, lat, lon}]}` |
| POST | `/api/jobs/{id}/whatif` | form: `activity`, `geo`, `legal`, `age` (multipliers 0-2, 1 = product setting), `top` (5-100, default 25), `evaluate` (bool). Returns `{multipliers, labels, items[{rank, company_id, name, municipality, default_rank, moved}], overlap_with_default{500,2000}, is_default, evaluation?}`. Default settings answer instantly, other settings in 2-6 s, `evaluate=true` adds ~7 s (Recall@500/2000 on the hidden-client benchmark with 4 splits: `evaluation{methods, lift_vs_baseline, repeats, note}`). |
| GET | `/api/enrichment/status` | `{available, source, fetched_on, looked_up, matched, match_rate, note}` for the optional Google Places enrichment (file `data/enrichment/places.json`, created once by `python run.py enrich`). |
| GET | `/api/enrichment/places?job=&limit=300` | pins for the map: `{source, items[{rank, company_id, name, lat, lon, google{place_id, name, rating, reviews, phone, website, maps_url, status, open_now, address, type, distance_m, named_query}}]}` |
| GET | `/api/lookalikes` | items now also carry `google` (object as above, or `null` when not fetched / not matched). Never used in scoring. |
| POST | `/api/jobs/{id}/chat` | JSON `{question (<=400 chars), ai_summary: bool}` -> `{question, text, kind: answer|info|help|refusal|fallback, rows[], columns[], sources[], filters, suggestions?, ai_generated:false, ai_summary{text,label}|null, llm_available, seconds}`. Read-only grounded assistant; 30 requests/min per job (429 after). The question is never sent to an LLM. |
| GET | `/api/transparency/{id}` | timings, models + fixed LLM options, k selection table, per-profile groundedness metadata, licenses, the four AI Act answers, limitations |
| GET | `/api/export/{id}/{name}` | `profiles.json`, `assignments.csv`, `lookalikes.csv` (one list, each business once; file order/score = global ranking, `rank` = unique global position 1..N, so `rank <= k` = first k rows), `lookalikes_by_profile.csv` (2000 per profile), `lookalikes_global.csv` (global ranking with name/municipality/vat), `assignments_detail.csv`, `analysis_meta.json` |

## Profile object
`profile_id, title, description, description_is_ai, description_meta{source: llm|mixed|template, verified, attempts, violations, model, options, ai_label, facts[], paragraphs[{key, title, text, is_ai, deterministic?, attempts, fact_ids}]}; `description` is the 3 paragraphs separated by a blank line, n_clients, n_client_vats, turnover_share (0-1 or null), stability (0-1), confidence (0-1), features[{feature, label, support, share_in_profile, share_in_universe, lift, shrunk_lift, p_value, q_value}]`

## UI rules (EU AI Act, brief section 9)
* Show `description_meta.ai_label` next to every description where `description_is_ai` is true.
* A description can be MIXED: show each `paragraphs[i]` with its own badge (AI vs 'Αυτόματο κείμενο από στατιστικά (χωρίς AI)'); `ai_label` describes the whole text.
* If `description_is_ai` is false the text is a deterministic template: do not call it AI-generated.
* Never invent numbers. Every figure shown must come from these fields.
* Never show a creditworthiness / solvency style score.
