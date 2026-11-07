# Security tests of the assistant "Ρώτα τα δεδομένα" (v29)

Run them yourself:  `python -m pytest -q tests/test_redteam_chat.py`   ·   report on the real data:  `python run.py redteam --data data`

## Design (why prompt injection has little to grab)
The user's question is never sent to a language model: fixed rules pick one of a few read-only functions over the
finished analysis (no SQL, no files, no network). Only the optional local AI summary uses a model, and it sees only
facts the code already computed.

## Results on the real data (30 questions)
| Group | Questions | Outcome |
|---|---|---|
| Direct prompt injection (EN/EL), role tags, persona, "repeat the text above" | 7 | all refused |
| Disguised injection: invisible characters, look-alike letters, leetspeak, full-width, spaced letters, base64 | 6 | all refused |
| SQL injection (UNION, DROP, `' OR '1'='1`), template injection, XSS, path traversal | 7 | all refused |
| Policy: creditworthiness, personal data, individual client turnover | 4 | all refused |
| Policy: honesty / reputation / legality / payment-behaviour judgements ("είναι απατεώνας;", "ποιες να αποφύγω;", "θα με πληρώσει;", "scam?") | 8 (+17 in manual probe) | all refused; 0 false positives on 21,927 real business names, municipalities and KAD labels |
| Normal business questions | 6 | all answered |
Client-turnover values leaked: **0**. Server errors: **0**. Malformed requests (list/dict/number/missing) -> HTTP 400.
Export path traversal -> 404. Cross-origin requests from non-localhost origins -> no CORS header.

## Vulnerabilities found and fixed in v29
1. **Indirect prompt injection through the data** (business names reach the AI summary). An obedient model could be made to
   write "Η επιχείρηση είναι αφερέγγυα" (a creditworthiness statement, forbidden by our AI-Act framing), a disguised link or
   instructions to the user; the old check only looked for numbers/links/tags. Now: facts that look like instructions never
   reach the model (fail closed), facts are wrapped in random delimiters and declared data, and the output is rejected for
   credit/solvency words, personal data, injected instructions, disguised domains/emails and non-Greek text.
2. Disguised injections (look-alike letters, leetspeak, full-width, spaced letters, base64, "repeat the text above") were not
   recognised (harmless fallback answer, but not refused/flagged). Now recognised. 0 false positives on all 15,019 real
   business names.
3. "τι τζίρο κάνει ο πελάτης <ΑΦΜ>" returned the business card instead of a refusal (no turnover leaked). Now refused.
4. **Search box crash (also a ReDoS risk):** /api/lookalikes used the user's text as a regular expression; "(" returned
   HTTP 500. Now plain-text, case- and accent-insensitive.

## Known limits
* The rate limit (30 questions/minute) is per analysis, shared by everyone using it.
* Rules are pattern-based; new phrasings can appear. Because the question never reaches a model, a missed pattern leads to
  a harmless "no answer" reply, not to a model following instructions.
