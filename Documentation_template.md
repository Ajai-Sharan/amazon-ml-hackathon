# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We treat entity resolution as *linking*: every Source 2 / Source 3 record is attached to at most one
Source 1 entity. A multi-key, IDF-weighted hashed blocking step proposes ≤10 Source 1 candidates per
record (recall ceiling ≈96–97 %). A LightGBM pairwise matcher scores them from ~43 string-similarity
and blocking-context features. A second LightGBM stage then re-scores each record's top links, adding
context from competing candidates and from the other records pointing at the same entity. The
decision threshold is tuned for macro F0.5 on out-of-fold predictions, giving a validation macro
F0.5 of **≈0.963**.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA on the 2.2 M / 5.0 M / 5.3 M training records:

* **Each Source 2/3 record matches at most one Source 1 entity.** 7.64 M ground-truth links point
  to 7.64 M distinct records, and 74 % of Source 2/3 records are matched. The rest are unrelated
  distractors. Entities have 0–11 matches (mode 3); 5.6 % are singletons.
* **Country never differs** between matched records, so blocking is done within a country label.
  `country` is otherwise treated as an opaque string, and France, which appears only in test, flows
  through the same code path.
* **Name noise:** legal-form variants (Pvt/Private, Ltd/Limited, L.L.C.), word-order shuffles
  ("OPX Limited Private India"), injected prefixes and suffixes ("M/s", "Center", "Partners",
  "-- ", "<<"), DBA names ("X doing business as Y"), website-style names ("dermatologycenter.com"),
  character typos and look-alike digits ("Antim0ny", "Federa1", "s1ate"), accents ("Súpreme"),
  and case changes.
* **Script noise:** about 5 % of Source 2/3 Indian names are written in Devanagari. Their vocabulary
  is tiny (172 distinct tokens), and state names appear in Devanagari, Telugu, Gujarati, Bengali
  and Kannada.
* **Address noise:** abbreviations (Rd/Road, R/Rue, BD/Boulevard), component reordering, missing
  components (≈3 % empty addresses), `<NULL>`/`N/A` placeholders, number perturbations (1951→1953,
  1083→1083B), state names vs. codes, and landmark phrases.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage gradient-boosted classifier + constrained assignment  
**Core Innovation:** Exploiting the "one Source 1 entity per record" structure. Queries are Source 2/3
records, each is assigned to its arg-max candidate, and a stage-2 model uses record-level and
entity-level competition features. We also learned a Devanagari→Latin token dictionary from the
training links.

---

## 3. Candidate Generation (Blocking)

**Normalisation** (`preprocess.py`):
* Devanagari names are translated token by token with a dictionary learned from training pairs.
  For matched pairs with equal token counts, tokens are aligned by position and the majority mapping
  kept; 19 generic shop words were added manually. Everything else is ASCII-folded with `unidecode`.
* Legal forms and filler words are canonicalised and removed to form the **core name**. DBA names
  are split into main and alternative names.
* Look-alike digits inside alphanumeric tokens are mapped back to letters, phone numbers are
  dropped, and a glued "com" suffix is stripped.
* A consonant-skeleton phonetic code is computed per token; it is robust to vowel typos and
  transliteration.
* Addresses: US state names map to codes, street-type and French or Indian abbreviations are
  canonicalised, ordinals are stripped, and digit groups are extracted.

**Blocking keys used** (`blocking.py`). All keys are prefixed by country and hashed to 64-bit:

| key | example |
|---|---|
| core name token | `sturdivant` |
| phonetic token | `strdvnt` |
| adjacent core-token pair (order-free) | `nu box` |
| first 3 core tokens (sorted) | `bhajanpura composites road` |
| full core name without spaces | `dermatologycenter` |
| 8-char prefix of that | `dermatol` |
| name token × house number | `katz#227` |
| DBA/alt-name token | `rizaveohaloio` |
| adjacent address-token pair | `white fence` |

Each key is weighted by `log(N / df)` times a per-type weight, where df is its frequency in
Source 1. Keys with df > 300 (address pairs: df > 60) are dropped, which bounds the join fan-out.
For every query we sum the weights of the keys it shares with each Source 1 record and keep the
**top 10** by score. The key tables are joined in Polars, in chunks, in separate processes to bound
memory.

* **Candidate pairs generated:** ≈99 M (train), ≈97.6 M (test); ~9.7 per Source 2/3 record, versus
  1.7 M × 10 M ≈ 1.7·10¹³ possible pairs, a reduction ratio above 0.999999.
* **How you ensured true matches were not lost:** many redundant key families let one clean token,
  number or address fragment suffice. Recall@K was measured on training data (recall@1 ≈ 0.93,
  @10 ≈ 0.966, @30 ≈ 0.977), and K = 10 was picked as the cost/recall knee.

---

## 4. Matching Model

**Stage 1 — pairwise LightGBM** (`features.py`, `model.py`), 43 features:
- **Name features:** Levenshtein ratio, token-set, token-sort, partial ratio and Jaro-Winkler on the
  core name. Also token-set and ratio on the full name, ratio and partial ratio on the no-space
  name, ratio and token-set on the phonetic skeleton, DBA-alternative similarities, token
  Jaccard/intersection, lengths and token counts.
- **Address features:** token-set, token-sort, ratio and partial ratio on the normalised address
  (missing when either side is empty). Also address-token Jaccard/intersection, house-number
  equality, containment, intersection count and fuzzy ratio, and address and number counts.
- **Other:** blocking score, number of shared keys, rank within the query's candidates, score
  relative to and gap from the best candidate, candidate count, and source (S2/S3).

**Stage 2 — context re-scoring** (`stage2.py`), on each query's top-3 stage-1 links:
- **Query side:** stage-1 probability and rank, best competing probability, margin, sum.
- **Entity side:** number of links pointing at the Source 1 entity, summed and maximum
  probability, the number of confident top-1 links overall and per source, this link's rank among
  them overall and within its source, and its probability relative to the entity's best link.

**Model type:** LightGBM gradient-boosted trees (MIT licence, a few MB, far below the 8 B
parameter limit). No pretrained or external models or data.  
**Training / validation:** queries are split into 2 folds by hash. Each stage-1 fold model is
trained on a sample of 900 k queries (~8.7 M pairs) and predicts the other fold, so every training
pair gets an out-of-fold score. Stage 2 is trained the same way on those out-of-fold scores, with
folds split by Source 1 entity.  
**Assignment and threshold selection method:** each query is linked to its arg-max stage-2 link when
the probability is at least *t*. *t* is chosen by sweeping and computing the exact challenge metric,
macro F0.5 over all 2.2 M training Source 1 entities, singletons included, on out-of-fold
predictions. The sweep gave t = 0.65. A per-entity expected-F0.5 top-k rule was also tried
(`postprocess.py`) but did not beat the threshold.

---

## 5. Results & Error Analysis

| version | validation macro F0.5 (OOF, full train) | P | R |
|---|---|---|---|
| v1: stage 1, threshold 0.7 | 0.9606 | 0.927 | 0.866 |
| v2: + stage-2 context model, threshold 0.65 | 0.9632 | 0.926 | 0.874 |
| v3: + normalisation/blocking fixes, more data | see README / final log | | |

(P and R are macro averages; singletons count P = R = 0 when correctly left empty and F = 1.)

- **Common false positives (wrong merges):** Near-duplicate Source 1 entities with the same name and
  a nearby house number (e.g. "Lucas, Harris & Reger Inc" at 2700 vs 2721 Allred Ave). Also records
  with an empty address whose name matches a different entity, and distractor records attached to
  the best available entity. Only ≈1.2 % of predicted links are wrong.
- **Common false negatives (missed matches):** ≈3.5 % of true links never reach the candidate set,
  mostly heavily corrupted names with an empty address. ≈3.6 % rank first but fall below the
  threshold: short generic names with a partial address, legal-form-only differences, or a changed
  house number. ≈1 % lose to a near-duplicate entity.

---

## 6. Conclusion
A cheap but redundant hashed blocking scheme with IDF weighting keeps ~97 % of true links. Pairwise
gradient boosting over classic string similarities then separates them well. The largest gains came
from the problem's structure: linking each record to at most one entity, and letting a second model
see the competition between candidates. The whole pipeline runs on a 4-core, 16 GB CPU machine in
about 2 hours and needs no external data or pretrained models.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
* `run_all.sh` — end-to-end entry point (data → blocking → matching → `output/`).
* `src/common.py` — paths, vocabularies, text folding, phonetic code.
* `src/build_translit.py` — Devanagari dictionary learned from training links.
* `src/preprocess.py` — name and address normalisation to parquet.
* `src/blocking.py` — hashed multi-key blocking, producing top-10 candidates per record.
* `src/features.py` — pairwise features.
* `src/model.py` — stage-1 LightGBM (2-fold OOF), threshold sweep, test scoring.
* `src/stage2.py` — stage-2 context model.
* `src/evaluate.py` — macro F0.5 metric.
* `src/make_submission.py` — writes `matching_results.tsv` and `candidate_pairs.tsv`.
* `src/postprocess.py` — expected-F0.5 rule (experimental, not used).

`candidate_pairs.tsv` is exactly the set of pairs scored by the stage-1 model, so every matched ID
is also a candidate.

### B. Additional Results
Blocking recall on a 3 % query sample, as a function of K: @1 0.929, @3 0.952, @5 0.959,
@10 0.966, @20 0.973, @30 0.977.

Stage-1 threshold sweep (OOF macro F0.5): 0.3 → 0.9501, 0.4 → 0.9558, 0.5 → 0.9589,
0.6 → 0.9603, 0.7 → 0.9606, 0.8 → 0.9591, 0.9 → 0.9521.

Stage-2 threshold sweep: 0.5 → 0.9613, 0.6 → 0.9628, 0.65 → 0.9632, 0.7 → 0.9631, 0.8 → 0.9620.

Most important stage-1 features (gain): blocking rank, relative blocking score, address token-set
ratio, first-house-number equality, blocking-score gap, full-name ratio.
