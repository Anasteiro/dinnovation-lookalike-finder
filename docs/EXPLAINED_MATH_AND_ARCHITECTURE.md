# How it works: two explanations of the same program

**Part 1** explains the maths the way you would explain it to a 10-year-old.
**Part 2** explains the architecture (data, database, local models) to a business owner, with enough detail that a computer
scientist can check every claim. Every number below comes from the real challenge data (24,424 businesses, 300 clients) and
can be reproduced with `python run.py evalproduct`.

---

# PART 1: The maths, like you're 10

## The story
Imagine you own an ice-cream shop and you have **300 regular customers**. The city has **24,000 other shops** that could
become customers. You can't visit all of them. Which 2,000 should you visit first?
The program answers this the way a clever friend would: *"Find out what your best customers have in common, then look for
shops that look just like them."* Here is how, step by step.

### Step 1: Turn every shop into a list of numbers (a "number card")
A computer can't understand "a taverna in Heraklion, set up as a limited company, two years old". So we describe each shop with
scores, like a trading card:
- **What it does** (the activity code, ΚΑΔ): we turn the *words* of the code into numbers, so "restaurant" and "taverna" end up
  with similar numbers even though the codes differ.
- **Where it is** (a smooth "map score": shops close together get similar scores).
- **How it is set up** (sole trader, limited company...) and **how old it is**.

### Step 2: "How alike are two shops?" = do their number cards point the same way?
Think of each number card as an arrow. Two arrows pointing the **same way** = very alike (score 1). Pointing **different ways**
= not alike (score near 0). Mathematicians call this *cosine similarity*; you can call it "the arrow test".

Not all parts matter equally, so we mix the four arrow tests like a recipe:
**activity counts most (1.0), then legal form (0.6), then location (0.3), then age (0.15).**
These recipe amounts were found by trying many mixes and keeping the one that found hidden customers best.

### Step 3: Don't believe everything people write
Shops *declare* their activity codes themselves. Some write 12 codes, like a person saying "I'm a footballer, a singer, a
pilot and an astronaut". Our rule: **the main code gets full trust (1.0); all the other codes together share a small
amount (0.5)**, and codes that make no sense next to the main one get less. So a shop with 12 codes is neither rewarded nor
punished.
We also give each shop a **cleanliness score**: `100 / (1 + unrelated codes / 3)`. 0 odd codes = 100, 3 odd codes = 50, 9 = 25.
It only tells you how tidy the paperwork is; it never changes the ranking.

### Step 4: Find the "types" of customer (grouping)
Put all 300 customers on a big table. Drop a few flags (group centres), let each customer join the nearest flag, move each flag
to the middle of its group, repeat until nothing moves. That is **k-means**. We give big customers a bit more say (a bigger
turnover pulls the flag harder).

*How many groups?* We **shake the table**: 30 times we remove random customers and regroup. If the same groups keep coming back,
they are real; if they change every time, they were an accident. The score for "same groups again" is called the
**Adjusted Rand Index** (1 = identical every time). On the real data the program keeps 2 groups, and each group
overlaps 95% and 93% with itself across the reshuffles (the "stability" shown for each profile).

### Step 5: "Is that special, or just luck?"
Say 170 of your customers are in one group, and the group has 20 shops from Kassandra. Kassandra holds only **0.6%** of all
shops. If you picked 170 shops blindly you'd expect about **1** from Kassandra, not 20.
The chance of that being luck is about **1 in 4 quintillion** (a 4 with 18 zeros): practically impossible. That chance is the **p-value** (computed with the hypergeometric
test, which is like pulling marbles out of a bag without putting them back).
We test hundreds of features, and if you buy hundreds of lottery tickets, winning once is no surprise. So we make the bar
higher the more things we test (**Benjamini-Hochberg**). Features that still pass are shown with their **lift**: "19 times more
common than in the market" (slightly shrunk when the group is small, so tiny groups can't brag).

### Step 6: Find the look-alikes using two judges
- **Judge A, the neighbours:** "Who are this shop's 5 closest customers? How alike are they?" Easy to understand.
- **Judge B, the pattern learner:** it learns a **scoring formula** that separates customers from everyone else. The clever
  part: the other 24,000 shops are not "bad", they are just *not yet known*. This is called **Positive-Unlabeled learning**: we
  tell the learner "these 300 are definitely good (and count each 8 times), the rest are unknown". It adjusts its formula
  150 times (a method called Adam on a *logistic regression*), like turning a dial until the hidden customers score high.
- **Final score = 20% Judge A + 80% Judge B.** We use each shop's *place in line* (percentile) instead of raw numbers, so the two
  judges can be mixed fairly.

### Step 7: Share the seats fairly
If customer type 1 brings 60% of your money and type 2 brings 40%, the top of the list should hold about 60% type-1 look-alikes
and 40% type-2. This is the **proportional budget merge**: each type gets seats in proportion to its customers (weighted by money),
and every shop appears **once**.

### Step 8: How do we know it works? Hide some customers!
We **secretly hide 20% of your real customers** (60 of 300), run everything on the other 240, and check how many of the hidden
ones show up in the top of our list. This is **Recall@k**: *found ÷ hidden*.

| | Top 500 | Top 2,000 |
|---|---|---|
| Our program | **15.3%** (±2.6) | **55.2%** (±1.6) |
| The competition's simple rule (same activity code) | 2.0% | 7.5% |
| Picking blindly | about 2.1% | about 8.3% |

Plain English: *following our first 2,000 suggestions finds more than half of the customers we hid; the simple rule finds
about as many as pure guessing.* The "±" says how much the number would wobble on another run (95% confidence).
(10 random hide-and-test rounds, `python run.py evalproduct --repeats 10`.)

---

# PART 2: The architecture: for business owners and computer scientists

## In one sentence
**A private, explainable "who should we sell to next" engine that runs entirely on one laptop, built on the public Greek
business register, and that shows its work.**

## Picture
```
 client list (Excel: VAT + turnover)      Greek business register (GEMI open data: 3 CSV, 24,424 businesses)
               \                                        /
                +-------------- ingest + clean --------+         (pandas; VAT normalised, shops matched by VAT)
                                   |
                       "number cards" for every business          (scikit-learn TF-IDF + SVD; NumPy / SciPy sparse)
                  activity | location | legal form | age | structure
                                   |
          +------------------------+---------------------------+
          |                        |                           |
   customer TYPES            statistics per type          look-alike RANKING
   (weighted k-means,        (Fisher exact test,          (kNN 20% + PU-learning 80%,
    stability by bootstrap)   Benjamini-Hochberg, lift)     proportional-budget merge)
          |                        |                           |
          +----------- facts (numbers) -------------+          |
                                   |                           |
              Greek description by the LOCAL language model     |
              (writes only from the facts; checked; labelled AI)|
                                   |                           |
        profiles.json   assignments.csv   lookalikes.csv  <----+     + web app on localhost:8000 (FastAPI + React)
                                   |
                    SQLite run history (every number traceable)
```

## The data
| Input | What it is |
|---|---|
| `vwHackCompany / CompanyActivity / Activities` | public GEMI registry: 24,424 active food-service businesses (ΚΑΔ 56), 96% with GPS |
| client Excel | 300 clients: VAT + turnover (turnover only *weights* customers; the chat never reveals an individual client's turnover) |

## The database
**There is no database server, on purpose.**
- **Why.** The whole market (24k rows) fits in memory as matrices (a few tens of MB). The maths is *exact* and takes seconds, so
  there is nothing to gain from an index or a server, and a server would break the rule "one command on a clean laptop".
- **SQLite** (built into Python, one file, no install, works offline) stores every analysis run with its settings and
  timestamp. This is the **audit trail** (EU AI Act traceability): any number on screen can be traced to the run that produced it.
- **Caches:** KAD text vectors and the PU-learning features are saved on disk, keyed by a fingerprint of the data, so a restart
  gets to the first result in about 4 s instead of 8 s, with identical results.
- **When to change it (honest):** a multi-user, multi-company product would move to **PostgreSQL + PostGIS** (shared data,
  permissions, geographic queries). That is a deployment step, not a change of algorithm.

## The models, and why each one
| Job | What we use | Why this one |
|---|---|---|
| Understand the *words* of activity codes | **TF-IDF over character n-grams + SVD (64 numbers)** by default; any local multilingual sentence model (e.g. multilingual-e5) can be plugged in | Only a few hundred *unique* texts, so it is instant and 100% offline and deterministic. Character n-grams cope with Greek word endings. The reported results use this default. (The neural option is supported; it is not needed to get the numbers above.) |
| Measure "alike" | **weighted cosine over 6 blocks** (activity 1.0, legal 0.6, geo 0.3, age 0.15, admin 0.1, structure 0.1) | One similarity used for grouping *and* ranking, so the answers agree with each other. Weights chosen by cross-validation on the real data. |
| Customer types | **weighted kernel k-means**, k from 1 to 6 chosen by bootstrap ARI with eligibility rules (stability ≥ 0.6, no tiny groups, real cohesion gain; otherwise "one profile") | Explainable, fast, and it *refuses* to invent groups that don't survive resampling. |
| "Is it special?" | **Fisher / hypergeometric test + Benjamini-Hochberg + shrunk lift** | The standard, defensible statistics for enrichment; protects against false discoveries. |
| Ranking | **PU-learning logistic regression (NumPy, Adam, 150 steps) + kNN**, merged by proportional budget | Only 300 positives against 24,000 unlabeled: a *convex*, regularised linear model is the right size (a neural net would overfit). Runs on CPU in about 1 s. 11 ranking variants were tested; none beat this beyond noise. |
| Greek wording | **EuroLLM-9B-Instruct (4-bit) through Ollama, temperature 0, seed 42** | EU-funded, Apache-2.0 licence, covers Greek, runs on 16 GB. Compared on licence, origin and cost against a Greek-specialised alternative (Llama-Krikri, Llama-3.1 licence); `run.py bakeoff` repeats the test. |
| Chat assistant | **Rules + whitelisted read-only functions; the user's text never reaches a model** | Prompt injection needs a model to follow instructions; here there is none to fool. The optional AI summary sees only computed facts and is rejected if it adds numbers, links, instructions or any creditworthiness wording. |

**The principle:** *the maths decides, the language model only writes.* The AI is never asked to judge a business; it
turns numbers the code already computed into readable Greek, and a checker rejects anything that adds a number or breaks a rule.

## Why it is efficient (what we measured, not what we hope)
- **Speed:** one full analysis of a client list ≈ **1.8 s** on a laptop CPU (no GPU); every data request answers in under 20 ms.
- **Accuracy:** Recall@2000 **55.2% ±1.6** and Recall@500 **15.3% ±2.6** on hidden real clients (baseline 7.5% / 2.0%).
- **Reproducible:** fixed seeds, same answer every run (`run.py consistency`).
- **Small:** peak memory for a full analysis is about **450 MB**; the only large download is the optional language model.
- **Language model speed:** about 9 s per profile on a 12 GB gaming GPU (RTX 3060). **Speed on the presentation laptop's CPU is
  not measured yet**: measure it before the demo (descriptions are cached, so the second run is instant, and without the model
  the app still works with automatic text, clearly labelled).
- **Honest scope:** "most efficient" means *best we measured among the variants we tested on this data*, not a universal claim.

## Security and compliance by design
- **Everything local:** no external AI or service is called while running; the website build is audited for external URLs;
  the server listens on localhost only. (Optional: Google Places pins, off unless you supply your own key; never stored.)
- **EU AI Act:** every AI-written paragraph is labelled; no creditworthiness or "trust" judgements about businesses
  (many are sole traders = natural persons), enforced in the writing checker *and* in the chat; full audit trail in SQLite.
- **Attack testing:** `python run.py redteam`: 20 prompt/SQL/XSS/path attacks, all refused, no client turnover leaked.

## What makes it different from market CRMs
This is **not** a replacement CRM (no pipeline, emails or contact data). It is the **prospecting brain** you plug in next to one.

| Typical CRM / sales-data tool | This program |
|---|---|
| AI runs in the vendor's cloud; your customer list is uploaded | **100% local**: your customer list never leaves the computer |
| Black-box "lead score" | **Explained**: each profile shows the features that define it with lift, p-value and stability; each suggestion has its reason |
| Rarely shows how accurate it is | **Built-in hold-out test** with Recall@k against a baseline, repeatable by the user |
| Generic across countries | Reads the **Greek registry (ΓΕΜΗ) and ΚΑΔ codes**, writes in **Greek** |
| Takes declared data at face value | Handles **noisy self-declared codes** (damping + cleanliness index) |
| Chatbots with cloud LLMs | Assistant where **no model ever reads the user's text**; red-teamed |
| Contact-level data (phones, owners) | Deliberately **no personal data**, no creditworthiness scoring |

## Limits (say them before a customer finds them)
- It measures **similarity to your customers**, not purchase intent or timing.
- Built on 300 clients and **food-service only**; other sectors need their own test.
- Public registry data only: no revenue, staff or behaviour for non-customers.
- The competitor comparison above describes typical products; check specific vendors before quoting them.
