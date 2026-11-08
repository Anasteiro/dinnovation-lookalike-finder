"""Part 6: the local LLM as a TRANSLATOR of validated statistics into Greek (requirement A6).

Pipeline per profile (v2, "small steps"):  numbered facts -> 3 short sections (who / what stands out / how to
approach), each written by the local model from ONLY 2-4 facts -> strict per-section verifier (numbers, novel terms,
forbidden words, repetition, Greek) -> one repair attempt -> deterministic sentence for that section if it still fails.
Citations [F#] are attached by CODE (the section's fact ids), not trusted to the model.
Every result carries the AI label (per paragraph) and the metadata needed for the model card / AI Act (Art. 50).
"""
from __future__ import annotations
import hashlib
import unicodedata
import json
import re
import time
import urllib.request
from pathlib import Path
from . import config as C

OLLAMA_URL = "http://localhost:11434"
GEN_OPTIONS = {"temperature": 0.0, "seed": 42, "top_p": 1.0, "num_predict": 200, "num_ctx": 2048}
PROMPT_VERSION = "v2"
# no creditworthiness / reliability / causal-claim language (AI Act trap + unsupported claims)
FORBIDDEN = ("πιστοληπτ", "φερέγγυ", "χρεοκοπ", "αξιόχρε", "αξιόπιστ", "αξιοπιστ", "κίνδυν", "ρίσκ", "καθιστ", "εγγυάται")

SYSTEM = ("Είσαι βοηθός που μεταφράζει στατιστικά αποτελέσματα σε απλά ελληνικά για πωλητές B2B. "
          "Δεν κάνεις δική σου ανάλυση και δεν προσθέτεις πληροφορίες.")


# ----------------------------------------------------------------------------- facts
def build_facts(profile: dict, max_features=6) -> list[dict]:
    """Numbered facts. Every claim the LLM may make must trace to one of these."""
    facts = []
    share = profile.get("turnover_share")
    t = f"Το προφίλ περιλαμβάνει {profile['n_clients']} πελάτες"
    t += f" που φέρνουν το {share:.0%} του συνολικού τζίρου." if share is not None else "."
    facts.append(t)
    facts.append(f"Σταθερότητα της ομάδας σε επαναδειγματοληψία (πόσο συχνά ξαναβρίσκεται η ίδια ομάδα πελατών): "
                 f"{profile['stability']:.0%}.")
    types = profile.get("business_types") or []
    if types:                                # what the clients are comes first in every description
        facts.append("Τύπος επιχειρήσεων του προφίλ: " + ", ".join(f"{t['label']} {t['share']:.0%}" for t in types[:3] if t["share"] >= 0.05) + ".")
    for f in profile.get("features", [])[:max_features]:
        facts.append(f"{f['label']}: {f['share_in_profile']:.0%} των πελατών του προφίλ έναντι "
                     f"{f['share_in_universe']:.0%} της αγοράς (φορές πάνω από το αναμενόμενο: {f['shrunk_lift']:.1f}).")
    return [{"id": f"F{i + 1}", "text": t} for i, t in enumerate(facts)]


def _short_label(fact_text: str) -> str:
    """'Κύριος ΚΑΔ: Ταβέρνες (56101200): 55% ...' -> 'Ταβέρνες'."""
    head = fact_text.split(": ", 1)
    name = head[1] if len(head) > 1 else head[0]
    return re.sub(r"\s*\(\d+\).*$", "", name.split(":")[0]).strip()


EXAMPLE_WHO = ("ΠΑΡΑΔΕΙΓΜΑ (με άλλα στοιχεία)\n[F1] Το προφίλ περιλαμβάνει 50 πελάτες που φέρνουν το 20% του συνολικού τζίρου.\n"
               "[F2] Κύριος ΚΑΔ: Υπηρεσίες ξενοδοχείων: 60% των πελατών του προφίλ έναντι 10% της αγοράς (φορές πάνω από το αναμενόμενο: 6.0).\n"
               "ΑΠΑΝΤΗΣΗ: Το προφίλ αποτελείται από 50 πελάτες, που φέρνουν το 20% του τζίρου. "
               "Οι περισσότεροι (60%) είναι ξενοδοχεία, ενώ στην αγορά το ποσοστό είναι μόλις 10%.")
def _activity_fact(facts: list[dict]) -> dict | None:
    """The fact describing the main activity (preferably a 'Κύριος ΚΑΔ' row)."""
    for f in facts[2:]:
        if f["text"].startswith("Κύριος ΚΑΔ"):
            return f
    return facts[2] if len(facts) > 2 else None


def build_sections(facts: list[dict]) -> list[dict]:
    """3 small writing jobs. Each one sees only its own facts (small prompt = far fewer model mistakes)."""
    by = {f["id"]: f for f in facts}
    top = [i for i in ("F3", "F4") if i in by]
    rest = [f["id"] for f in facts[4:6]]
    return [
        {"key": "who", "title": "Ποιοι είναι", "ids": ["F1"] + top, "numbers": True, "max_sentences": 3, "example": EXAMPLE_WHO,
         "task": "Γράψε 1-2 σύντομες προτάσεις που περιγράφουν ποιοι είναι οι πελάτες αυτού του προφίλ "
                 "(πόσοι είναι, τι μερίδα του τζίρου φέρνουν, ποια είναι η κύρια δραστηριότητά τους)."},
        {"key": "stands_out", "title": "Τι τους ξεχωρίζει", "ids": ["F2"] + rest, "numbers": True, "max_sentences": 3,
         "example": "",
         "task": "Γράψε 2 σύντομες προτάσεις. Στην πρώτη πες πόσο σταθερά ξαναβγαίνει η ίδια ομάδα πελατών όταν η ανάλυση "
                 "επαναλαμβάνεται (το πρώτο στοιχείο). Στη δεύτερη πες ποιο χαρακτηριστικό τους ξεχωρίζει από την αγορά "
                 "(τα υπόλοιπα στοιχεία). Αντέγραψε τα ποσοστά ΑΚΡΙΒΩΣ όπως είναι γραμμένα. "
                 "Η «σταθερότητα» ΔΕΝ σημαίνει ότι οι πελάτες είναι σταθεροί ή αξιόπιστοι."},
        {"key": "approach", "title": "Πώς να τους προσεγγίσει ο πωλητής", "ids": [f["id"] for f in facts[2:]] or ["F1"],
         "numbers": False, "max_sentences": 1, "deterministic": True},
    ]


def build_prompt(section: dict, facts: list[dict], extra: str = "") -> str:
    by = {f["id"]: f["text"] for f in facts}
    lines = "\n".join(f"[{i}] {by[i]}" for i in section["ids"])
    num_rule = ("Χρησιμοποίησε μόνο αριθμούς που υπάρχουν στα στοιχεία." if section["numbers"]
                else "Μην γράψεις κανέναν αριθμό.")
    ex = (section.get("example") or "") + ("\n\nΤΩΡΑ ΕΣΥ\n" if section.get("example") else "")
    return (f"{section['task']}\n"
            "ΚΑΝΟΝΕΣ: Χρησιμοποίησε ΜΟΝΟ τα παρακάτω στοιχεία. Μην προσθέσεις πληροφορίες, εκτιμήσεις ή αιτιολογήσεις που δεν "
            f"υπάρχουν σε αυτά. {num_rule} Μην γράψεις παραπομπές, τίτλους ή λίστες. Μόνο ελληνικά. "
            "Μη μιλήσεις για αξιοπιστία, πιστοληπτική ικανότητα ή οικονομική υγεία.\n\n"
            f"{ex}ΣΤΟΙΧΕΙΑ:\n{lines}\n{extra}ΑΠΑΝΤΗΣΗ:")


# ----------------------------------------------------------------------------- verifier
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[Α-Ωα-ωάέήίόύώϊϋΐΰΆΈΉΊΌΎΏ]{5,}")
_GREEK = re.compile(r"[Α-Ωα-ωάέήίόύώϊϋΐΰ]")
_LATIN = re.compile(r"[A-Za-z]")

# generic Greek words the model may use freely (stems, 5 letters, accents/case removed)
_ALLOWED_WORDS = """πελατ προφιλ αποτελ περιλαμ ποσοστ αγορα επιχειρ συγκεντρ σημαντ υψηλο περισσ συνηθω ιδιαιτ κυριω
πωλητ προσεγγ επικοιν προτασ προσφορ εξυπηρ αναγκε δραστηρ χαρακτη αναμενο επαναδε σταθερ τζιρου πανω εναντι γενικα ομαδα
κατηγορ τομεα κλαδο κλαδου αποτελο ανηκουν ανηκει εχουν εχει ειναι υπαρχει υπηρεσ ξεχωριζ διαφερ μεγαλυτ μικροτ μεγαλο
μικρο συμμετο παρουσι παρατηρ φερνουν φερνει συνολικ τζιρος τζιρο ισχυρη ισχυρο εντονη εντονο εστιασ εστιαζ ωστοσο ενω
ταυτοχρ επισης επιπλεον μονο μολις περιπου σχεδον παραπανω παρακατω σχετικ προσαρμο προσαρμοσ αντιστοιχ μορφη μορφης
δηλαδη εξηγει ανεβαζ μεσα ανα αναμεσ οπως οταν εκει ολοκληρ κοινα κοινο κοινη ποιοτητ δυνατ δυνατο ευκαιρ εφοσον κατα
ενδιαφερ συγκεκριμ ανταποκρ επικεντρ προωθηση πρωτα πρωτη αρχικα ξεκιν ξεκινησ ανακοιν παρουσιασ ενημερωσ προτειν
συνεργασ δηλων δηλωμ συγκρι σχετικ μπορει τομεισ τομεας δυναμ γνωστ αγορες αγορας ποσοτ ποσοστο ομαδες ομαδας γεωγραφ περιοχ""".split()
def _norm(w: str) -> str:
    w = unicodedata.normalize("NFD", w.lower())
    w = "".join(c for c in w if not unicodedata.combining(c)).replace("ς", "σ")
    return w


def _stem(w: str) -> str:
    return _norm(w)[:5]


_ALLOWED = {_stem(w) for w in _ALLOWED_WORDS}
_PEOPLE_STABLE = re.compile(r"σταθεροί|σταθερούς|σταθερών|σταθερές")      # stability is a property of the GROUP, not the clients


def _nums(s: str) -> list[float]:
    return [float(x.replace(",", ".")) for x in _NUM.findall(s)]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]


def strip_citations(text: str) -> str:
    t = re.sub(r"\s*(?:\[F[^\]]*\])+", "", text)
    return re.sub(r"[ \t]+", " ", t).replace(" .", ".").replace(" ,", ",").strip()


def novel_terms(text: str, fact_texts: list[str]) -> list[str]:
    """Content words (>=6 letters) whose 5-letter stem is neither in the facts nor in the generic allow-list.
    Catches invented claims ('ιδιοκτήτες') and invented places ('Πάτρας'). A capitalised word in mid-sentence
    (a name) that is not in the facts counts double."""
    known = {_stem(w) for t in fact_texts for w in _WORD.findall(t)} | _ALLOWED
    out, seen = [], set()
    for sent in _sentences(text):
        for i, m in enumerate(_WORD.finditer(sent)):
            w = m.group(0)
            if len(w) < 6 or _stem(w) in known or _stem(w) in seen:
                continue
            seen.add(_stem(w))
            out.append(w)
            if i > 0 and w[0].isupper():          # proper name not in the facts
                out.append(w + "*")
    return out


def verify_section(text: str, facts: list[dict], section: dict, max_novel=1) -> list[str]:
    """Violations for one paragraph (empty = grounded). Checks: Greek; <=N sentences; no repeated sentence; no
    forbidden words; numbers only from the section's own facts (none at all for the advice paragraph); at most
    `max_novel` words that do not come from the facts. It does NOT prove semantic correctness of every word -
    stated as a limitation in the model card."""
    by = {f["id"]: f["text"] for f in facts}
    ft = [by[i] for i in section["ids"]]
    bad = []
    t = text.strip()
    if not t:
        return ["κενή απάντηση"]
    if re.search(r"\[F\d+", t):
        bad.append("περιέχει παραπομπές (οι παραπομπές μπαίνουν αυτόματα)")
    letters = len(_GREEK.findall(t)) + len(_LATIN.findall(t))
    if not letters or len(_GREEK.findall(t)) / letters < 0.9:
        bad.append("το κείμενο δεν είναι κυρίως ελληνικό")
    low = t.lower()
    bad += [f"απαγορευμένος όρος: {w}" for w in FORBIDDEN if w in low]
    if _PEOPLE_STABLE.search(low):
        bad.append("η σταθερότητα αφορά την ομάδα, όχι τους πελάτες")
    sents = _sentences(t)
    if len(sents) > section["max_sentences"]:
        bad.append(f"πάνω από {section['max_sentences']} προτάσεις")
    if len({_norm(s) for s in sents}) < len(sents):
        bad.append("επανάληψη πρότασης")
    nums = _nums(t)
    if not section["numbers"] and nums:
        bad.append("αριθμοί σε παράγραφο χωρίς αριθμούς")
    allowed = [n for x in ft for n in _nums(x)]
    for n in nums if section["numbers"] else []:
        if not any(abs(n - a) <= 0.051 or round(a) == n for a in allowed):
            bad.append(f"αριθμός {n:g} δεν υπάρχει στα στοιχεία")
    nov = novel_terms(t, ft)
    if len(nov) > max_novel:
        bad.append("λέξεις/έννοιες εκτός στοιχείων: " + ", ".join(w.rstrip("*") for w in nov if not w.endswith("*"))[:80])
    return bad


# ----------------------------------------------------------------------------- generation
def ollama_generate(model: str, host: str = OLLAMA_URL, options: dict | None = None, timeout=300):
    """Returns generate(prompt, system) -> (text, stats). Uses only the local Ollama server."""
    opts = dict(GEN_OPTIONS if options is None else options)

    def gen(prompt: str, system: str = SYSTEM):
        body = json.dumps({"model": model, "stream": False, "keep_alive": "20m", "options": opts,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(host + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        n, dur = d.get("eval_count", 0), d.get("eval_duration", 0) / 1e9
        return d["message"]["content"].strip(), {"seconds": round(time.time() - t0, 2), "tokens": n,
                                                  "tokens_per_s": round(n / dur, 1) if dur else None}
    gen.model, gen.options = model, opts
    return gen


_LABEL = re.compile(r"^(.*?): \d+%")


def targeting_rule(facts_subset: list[dict]) -> str:
    """Deterministic advice paragraph built from the significant features (no AI, cannot hallucinate)."""
    act = place = legal = None
    for f in facts_subset:
        m = _LABEL.match(f["text"])
        if not m:
            continue
        lab = m.group(1)
        if lab.startswith("Κύριος ΚΑΔ: ") and not act:
            act = re.sub(r"\s*\(\d+\)$", "", lab.split(": ", 1)[1])
        elif lab.startswith("Δήμος ") and (not place or place.startswith("στον Νομό")):
            place = "στον Δήμο " + lab.split(" ", 1)[1]
        elif lab.startswith("Νομός ") and not place:
            place = "στον Νομό " + lab.split(" ", 1)[1]
        elif lab.startswith("Νομική μορφή ") and not legal:
            legal = lab.split("Νομική μορφή ", 1)[1]
    parts = []
    if act:
        parts.append(f"με κύρια δραστηριότητα «{act}»")
    if place:
        parts.append(place)
    if legal:
        parts.append(f"με νομική μορφή {legal}")
    if not parts:
        return "Ο πωλητής μπορεί να ξεκινήσει από επιχειρήσεις που μοιάζουν με τους πελάτες αυτού του προφίλ."
    return "Προτεινόμενη στόχευση για τον πωλητή: επιχειρήσεις " + ", ".join(parts) + "."


def section_template(section: dict, facts: list[dict]) -> str:
    """Deterministic sentence(s) for a section - no AI involved, so NOT labelled as AI."""
    by = {f["id"]: f for f in facts}
    if section["key"] == "approach":
        return targeting_rule([by[i] for i in section["ids"]])
    return " ".join(by[i]["text"] for i in section["ids"])


def template_text(profile: dict, facts: list[dict]) -> str:
    return "\n\n".join(section_template(s, facts) for s in build_sections(facts))


def _agg(stats_list: list[dict]) -> dict:
    sec = round(sum(s.get("seconds", 0) for s in stats_list), 2)
    tok = sum(s.get("tokens", 0) for s in stats_list)
    t_gen = sum(s["tokens"] / s["tokens_per_s"] for s in stats_list if s.get("tokens_per_s") and s.get("tokens"))
    return {"seconds": sec, "tokens": tok, "tokens_per_s": round(tok / t_gen, 1) if t_gen else None}


def write_section(section: dict, facts: list[dict], generate, max_attempts=2):
    """Returns (text, is_ai, attempts, violations, stats)."""
    extra, viol, stats, text = "", [], [], ""
    for a in range(1, max_attempts + 1):
        text, st = generate(build_prompt(section, facts, extra))
        stats.append(st or {})
        text = re.sub(r"^\s*(ΑΠΑΝΤΗΣΗ|Απάντηση)\s*:\s*", "", text.strip())
        viol = verify_section(text, facts, section)
        if not viol:
            return text, True, a, [], _agg(stats)
        extra = "ΠΡΟΣΟΧΗ: η προηγούμενη απάντηση παραβίασε τους κανόνες: " + "; ".join(viol[:3]) + ". Διόρθωσέ το.\n"
    return section_template(section, facts), False, max_attempts, viol + ["ΑΠΟΡΡΙΦΘΗΚΕ: " + text[:300]], _agg(stats)


def narrate(profile: dict, generate, cache_dir: Path | None = None, max_attempts=2) -> dict:
    facts = build_facts(profile)
    sections = build_sections(facts)
    model = getattr(generate, "model", "unknown")
    opts = getattr(generate, "options", GEN_OPTIONS)
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, model, opts, facts], sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()[:20]
    cpath = (cache_dir / f"narr_{key}.json") if cache_dir else None
    if cpath and cpath.exists():
        return {**json.loads(cpath.read_text(encoding="utf-8")), "cached": True}
    parts, all_stats = [], []
    for s in sections:
        if s.get("deterministic"):
            parts.append({"key": s["key"], "title": s["title"], "fact_ids": s["ids"], "is_ai": False, "attempts": 0,
                          "violations": [], "text": section_template(s, facts), "deterministic": True})
            continue
        text, ok, att, viol, st = write_section(s, facts, generate, max_attempts)
        all_stats.append(st)
        parts.append({"key": s["key"], "title": s["title"], "fact_ids": s["ids"], "is_ai": ok, "attempts": att,
                      "violations": viol, "text": text})
    n_ai = sum(p["is_ai"] for p in parts)
    verified = all(p["is_ai"] for p in parts if not p.get("deterministic"))
    cited = "\n\n".join(p["text"] + " " + "".join(f"[{i}]" for i in p["fact_ids"][:4]) for p in parts)
    clean = "\n\n".join(p["text"] for p in parts)
    if verified and n_ai == len(parts):
        label = f"Κείμενο που παρήχθη από AI ({model}, τοπική εκτέλεση)"
    elif n_ai:
        idx = ", ".join(str(i + 1) for i, p in enumerate(parts) if p["is_ai"])
        label = (f"Οι παράγραφοι {idx} παρήχθησαν από AI ({model}, τοπική εκτέλεση)· οι υπόλοιπες είναι αυτόματο "
                 "κείμενο από στατιστικά (χωρίς AI)")
    else:
        label = "Αυτόματο κείμενο από στατιστικά (χωρίς AI)"
    out = {"profile_id": profile["profile_id"], "source": ("llm" if n_ai == len(parts) else "mixed") if verified and n_ai else ("mixed" if n_ai else "template"),
           "is_ai": n_ai > 0, "verified": verified, "attempts": max(p["attempts"] for p in parts),
           "violations": [f"{p['title']}: {v}" for p in parts for v in p["violations"]],
           "text": cited, "text_clean": clean, "paragraphs": parts, "facts": facts, "model": model, "options": opts,
           "prompt_version": PROMPT_VERSION, "facts_hash": key, "stats": _agg(all_stats), "ai_label": label,
           "cached": False}
    if cpath:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def narrate_profiles(profiles: list[dict], generate, cache_dir: Path | None = None, on_progress=None) -> list[dict]:
    """Fills description / description_is_ai / ai_label of every profile (A6)."""
    for i, p in enumerate(profiles):
        r = narrate(p, generate, cache_dir)
        if on_progress:
            on_progress(i + 1, len(profiles), r)
        p["description"] = r["text_clean"]
        p["description_is_ai"] = r["is_ai"]
        p["description_meta"] = {k: r[k] for k in ("source", "verified", "attempts", "violations", "model", "options",
                                                    "facts_hash", "stats", "ai_label", "facts", "paragraphs", "prompt_version")}
    return profiles
