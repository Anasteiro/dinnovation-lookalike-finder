"""Central configuration. Everything tunable lives here or in CLI flags."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = ROOT / "outputs"

SEED = 42
KS = (500, 2000)                       # Recall@k from the brief
REF_DATE = pd.Timestamp("2026-11-06")  # reference date for company age

# KAD weighting (A1/A3): primary counts fully, ALL other declared codes share a fixed mass,
# so a business declaring 12 codes is neither inflated nor punished.
PRIMARY_WEIGHT = 1.0
OTHER_TOTAL = 0.5
AUX_FACTOR = 0.5                       # auxiliary codes count half of secondary ones

# Similarity block weights, tuned by cross-validation on the real challenge data (scripts/tune_cv.py, stages 2-4):
# less weight on administrative one-hots (geo grid already carries location), more on legal form.
DEFAULT_BLOCK_WEIGHTS = {
    "activity": 1.0,
    "admin": 0.1,
    "geo": 0.3,
    "legal": 0.6,
    "age": 0.15,
    "struct": 0.1,
}
# Lookalike score = mean similarity to the m nearest clients (pure kNN). A centroid component HURT recall on real data.
KNN_M = 5
CENTROID_BLEND = 0.0

# Lookalike ranking = per-profile percentile blend of kNN similarity and PU-learning (idea from the second developer), merged by proportional budget
RANK_W_KNN = 0.2
RANK_W_PU = 0.8

NOISE_DAMP = True      # damp implausible declared secondary KADs (idea from the second developer); set False to compare
