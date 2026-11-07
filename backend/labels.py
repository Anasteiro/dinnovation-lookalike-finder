"""Readable short Greek names for activity codes and legal forms (used in profile titles and statistics labels)."""
from __future__ import annotations
from . import ingest as I

SHORT_KAD = [  # readable short names for the activity codes of the food-service sector (checked on the description text)
    (("ΣΧΟΛΙΚ",), "Σχολικά κυλικεία"), (("ΚΥΛΙΚΕΙ",), "Κυλικεία"), (("ΚΑΝΤΙΝ",), "Καντίνες"),
    (("ΚΙΝΗΤ",), "Κινητές μονάδες εστίασης"), (("ΤΡΟΦΟΔΟΣΙ",), "Τροφοδοσία (catering)"), (("CATERING",), "Τροφοδοσία (catering)"),
    (("ΠΛΗΡΗ ΕΞΥΠΗΡΕΤΗΣΗ",), "Εστιατόρια"), (("ΕΣΤΙΑΤΟΡ",), "Εστιατόρια"), (("ΤΑΒΕΡΝ",), "Ταβέρνες"), (("ΠΙΤΣ",), "Πιτσαρίες"),
    (("ΨΗΤΟΠΩΛ",), "Ψητοπωλεία"), (("ΠΕΡΙΟΡΙΣΜΕΝΗ ΕΞΥΠΗΡΕΤΗΣΗ",), "Ταχυφαγεία"), (("ΤΑΧΥΦΑΓ",), "Ταχυφαγεία"), (("ΣΝΑΚ",), "Σνακ μπαρ"),
    (("ΚΑΦΕΤΕΡΙ",), "Καφετέριες"), (("ΚΑΦΕΝΕΙ",), "Καφενεία"), (("ΚΑΦΕ",), "Καφέ"), (("ΑΝΑΨΥΚΤΗΡΙ",), "Αναψυκτήρια"),
    (("ΜΠΑΡ",), "Μπαρ"), (("ΠΟΤΩΝ",), "Μπαρ και ποτά"), (("ΖΑΧΑΡΟΠΛΑΣΤ",), "Ζαχαροπλαστεία"), (("ΑΡΤΟΠΟΙ",), "Αρτοποιεία"),
]
SHORT_LEGAL = {"ΑΤΟΜΙΚΗ": "Ατομικές επιχειρήσεις", "ΙΚΕ": "ΙΚΕ", "ΟΕ": "Ομόρρυθμες (ΟΕ)", "ΕΕ": "Ετερόρρυθμες (ΕΕ)", "ΑΕ": "Ανώνυμες (ΑΕ)",
               "ΕΠΕ": "ΕΠΕ"}


def short_kad(desc: str) -> str:
    d = I.strip_accents(desc).upper()
    for keys, label in SHORT_KAD:
        if all(k in d for k in keys):
            return label
    t = desc.split(" (")[0].strip()
    t = t[:1].upper() + t[1:].lower()
    return t if len(t) <= 40 else t[:40].rsplit(" ", 1)[0] + "…"


