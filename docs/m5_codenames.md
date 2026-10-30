# M5 design: the made-up names test (v4: A4+R benchmark, entity split)

Status: committed before any swapped-corpus run, which is what pre-registers the expectation below. The entity list
(`configs/m5_entities.csv`) was approved as is and frozen with `thedu freeze --add` (2026-10-28). M5 runs after the A4+R runs finish. Rules: see the 2026-10-28 journal decision (embeddings are the
benchmark, not a component; keyword index only, all intelligence at query time).

## Question

Embeddings know real names (Tesla, Vanguard). BM25 only sees tokens. When names become invented codenames, does the
embedding advantage over embedding-free pipelines shrink, and do those pipelines catch A4?

## Arms

| Arm | What it is | Role |
| --- | --- | --- |
| A0 | BM25 | embedding-free |
| A3 | BM25 with grounded rewrite | embedding-free |
| A0+R | A0 top 50, reranked by bge-reranker-v2-m3 (zero-shot, not tuned) | embedding-free |
| A3+R | A3 top 50, reranked by the same model | embedding-free |
| A0C | BM25 over words plus character 3 to 5-grams inside words ("chars-plus" in the index sweep), no stemming for the grams | embedding-free, extra arm |
| A4 | Qwen3-Embedding-0.6B | benchmark |
| A4+R | A4 top 50, reranked by the same cross-encoder | benchmark: a reranker is a standard RAG stage |
| H0, A5, frozen router | as before | context only, not tested |

Every embedding-free arm is compared with both benchmarks. A0 and A3 are compared with A4; the reranked arms A0+R and A3+R are
also compared with A4+R (same reranker on both sides).

A0C is there because a codename is an unknown word: character pieces can still match a codename that is spelled the same in
question and documents, and in the index sweep it was the only variant that helped anywhere (NFCorpus, +0.029; it lost on
SciFact and FiQA). It is rebuilt on each swapped corpus. Reranker depth is 50 at every level and is not tuned on swapped data. Reranking runs at
query time only; the index is a plain keyword index (strict rules). Rerank cost is priced at the reference rate in
`configs/pricing_rerank.yaml`; rewrite cost as before.

## Data and the swap

- **Corpus and questions:** FiQA, dev questions only (500). Test answers are not read. The corpus is shared across splits,
  so swapped docs are rebuilt for the whole corpus; qrels and doc ids never change.
- **Entities (54, not 200):** FiQA dev questions mention few named things; they are mostly short, generic money questions.
  Candidates come from a rule (capitalised phrases and tickers in the documents, matched case-insensitively against the
  500 dev questions, at least 3 documents each), then the pinned qwen3-14b (through the gateway, cached) keeps proper
  names, merges aliases and writes a one-line plain description. I then merged duplicates (United States / America / USA),
  corrected one wrong answer (BMW had come back as "Bayer AG") and rewrote 20 descriptions so none contains the name of any
  swapped entity; each hand edit is in the `note` column of **`configs/m5_entities.csv`**. 54 rows are `keep=y`: companies,
  agencies, exchanges, funds, countries and places, a person, named account types (IRA, Roth IRA, ISA, RRSP, TFSA, SIPP),
  currencies, websites. 26 rows are `keep=n` as generic terms rather than names (IPO, GDP, VAT, LLC, free cash flow, tax
  return, ...); flip them to `y` if you want them, which gives 80 entities, 119 affected questions and 16,002 documents.
  The reviewed list is frozen (`thedu freeze --add`) before any run.
  With the 54: **92 of 500 dev questions** contain at least one (76 one, 14 two, 2 three) and **14,647 of 57,638 documents**
  change at 100% (US, IRS, IRA appear in thousands). With 92 questions the paired ranges are wide (see the primary test).
- **Codenames:** one per entity, pronounceable (Valdrix), generated from a seed, never a word or token in the corpus or
  questions. All aliases map to the same codename. Whole-word, case-preserving, possessives and `$TICKER` forms handled,
  longest alias first.
- **Glossary:** one short document per swapped codename (54 at 100%) ("`<codename>`: `<description>`"), each well under 512 tokens, so
  no truncation asymmetry. Glossary documents are relevant to no question; they can only help a rewriter that retrieves
  them, or displace real answers in a ranking (see mechanism notes).
- **Residual audit:** after swapping, count original names left in docs and questions, per level. If more than 2% of
  occurrences remain, fix the alias list and rebuild before running.

## Conditions and sweep

| | Question | Documents |
| --- | --- | --- |
| **A** | codename | codename |
| **B** | plain description (the check-list text) | codename |

Level = 0, 25, 50, 100% of the entities swapped, nested (25 ⊂ 50 ⊂ 100, seeded shuffle). 0% is the original FiQA for both
conditions, no glossary. A0, A3, A4 at 0% reuse run `890ad993c9` (same pinned provider and questions); the rerank arms
at 0% are new runs.

**Evaluated on** the 92 questions containing at least one of the 54 entities (fixed across levels and conditions, so every
comparison is paired). The count is reported first; under 100 questions the ranges are wide and the post says so. The full
500 are context. Only documents that change are re-embedded (about 14,600 at 100%, roughly 20 minutes at the measured 12 documents per second; fewer at 25% and 50%). Rewrites and reranks run on the affected questions only. The
budget guard stays on.

## Primary test and expectation (pre-registered)

All differences are paired on the affected questions, 95% bootstrap range. "Gap" means arm minus A4.

**Primary tests.** All at the 100% level, asked once per condition (A and B), verdict reported for each; the expectation is
supported only if it holds in both. "Match" = the lower bound of the gap (arm minus benchmark) is above -0.01; "beat" = the
lower bound is above 0.

- **P1:** does any embedding-free arm (A0, A3, A0+R, A3+R, A0C) match or beat A4?
- **P2:** does A0+R match or beat A4+R?
- **P3:** does A3+R match or beat A4+R?

Seven comparisons per condition with no correction, so a single marginal pass is weak evidence and the post says so. Only 92
questions are affected, so a range half-width of 0.04 to 0.06 is likely: a miss counts as "the benchmark is better" only
when the upper bound of the gap is below 0; otherwise it is reported as inconclusive.

**Expectation.** Embeddings lose their advantage as names become codenames, and rewriting, reranking and character
n-grams help the keyword side more than they help A4: at least one of P1, P2 or P3 passes in condition A **and** in condition B (the expectation holds only if it passes in both).

**Secondary tests** (reported, labelled secondary, none of them decides the verdict):

- **S1:** A4 nDCG@10 falls, 100% level minus 0% level: upper bound below 0, in A and in B.
- **S2:** the gap of each embedding-free arm rises, 100% minus 0%: lower bound above 0, in A and in B. Fifteen tests with no correction.
- **S3 (entity split, pre-registered):** P1 to P3 repeated on two groups of affected questions: questions whose entities are
  all **organisations and products** (types company, agency, exchange, fund, index, account, website, product, person, other:
  37 entities) and questions whose entities are all **places and currencies** (country, region, place, city, currency: 17
  entities). Questions with both kinds are reported separately and in neither group. Countries carry many of the 92 questions,
  and internal-name scenarios are closer to organisations and products. Group sizes are reported first; small groups give
  inconclusive ranges.
- The lowest swap level at which each arm first matches A4; each arm's cost per search against A4's; intermediate levels.

Mechanism, stated now so surprises are visible (not criteria):

- **A0 should not move in condition A.** A consistent mapping preserves every token's frequency, so S2 for A0 is mostly A4
  falling. Any A0 change beyond ±0.01 means the swap altered something besides names (multi-word names, glossary); I check
  that before trusting the rest.
- **A3 may not gain by itself.** The rewriter loses its knowledge of the entity too (no "Tesla → TSLA"). The expected gain
  is relative to A4.
- **The reranker is a model with world knowledge as well.** In A the codename appears in both question and document, so
  exact overlap survives and it may hold up; in B it sees a description against a codename and cannot bridge them unless a
  glossary document is among the candidates. The rerank arms may fall in B.
- **Glossary documents are distractors in B.** A description-only question matches its own glossary entry almost perfectly
  for BM25 and for embeddings, and that document is relevant to nothing. Expect some displacement at the top ranks; the
  rewrite arms can use it as a bridge (the first-round top 5 shows the codename).
- **A0C should hold in A.** The grams of a codename match the grams of the same codename in the documents, so A0C should
  move little in A; in B the question has no codename, so it has nothing to match.
- Candidate depth is a separate limit: Phase A showed 30% of FiQA questions have no relevant document in BM25's top 100.

## Tests to build with it

Mapping consistent and invertible; longest alias first; case and possessive handling; qrels and doc ids unchanged; 0%
level byte-identical to the original; same seed gives the same swap; no codename occurs in the original corpus; residual
audit counts correctly; check list refuses to freeze with blank codenames or descriptions containing a name; reranker only
reorders its candidates (same set, same length, order changes only).

## Settled points

- Both conditions are kept; the expectation holds only if it passes in both (approved 2026-10-28).
- Condition B swaps the question and the docs for the same entities; entities not swapped keep their real name in both.
- The entity list is approved and frozen; the 26 generic terms stay `keep=n`.
- Cost to run: the rerank arms (A0+R, A3+R, A4+R) take about 20 minutes per level and condition (92 questions), seven runs,
  plus about an hour of re-embedding; rewrites are cheap.

## Amendment 2026-10-28 (before any M5 run): demonyms, plurals, one more secondary split

The residual audit of the first swap build failed the 2% rule: 9.4%, 10.3% and 7.7% of name occurrences were left at the 25, 50 and
100% levels, nearly all demonyms and plurals (American 1,276, Indian 456, Australian 173, Europeans, Pakistani, TFSAs). The frozen
list `configs/m5_entities.csv` is not edited. The amendment is a second file, **`configs/m5_aliases_2026-10-28.csv`**, frozen with
`thedu freeze --add`, and it wins where it names the same word.

- **Demonyms** map to a derived form of the codename, never the bare codename: American to `<codename>ian`, Americans to
  `<codename>ians`, Indian to `<codename>ian` (Chinese, Canadian, British, Brit, Australian, European, Pakistani, Singaporean
  the same way; New Yorker to `<codename>er`). **Plurals** map to codename + `s` (IRAs, Roth IRAs, ISAs, RRSPs, TFSAs, SIPPs,
  BMWs, USDs, Chinas, ...). The frozen list had mapped `Americans`, `European`, `IRAs` and `RRSPs` to the bare codename; the
  amendment replaces those. The rows are listed in the file (45 forms, one exact-case rule).
- **Boundaries:** whole words only. Indiana is not India, and Americas, Americana and `nyse.com` URLs are not touched where the
  word does not match whole. `US` and the other acronyms still match only in capitals (`us` stays a pronoun). `Trading Places`
  matches only as written (the lower-case phrase is ordinary English). Acronym case is kept as it was: an acronym becomes an
  upper-case codename (`IRA` to `ROKUGUR`), a plural becomes `Rokugurs`.
- **Condition B is not smoothed:** a demonym or plural in a question is replaced by the entity's description as it stands.
- **Britain** is an alias of the United Kingdom (bare codename, like the other aliases; added on review). **English** (the language) is left unswapped.
- **Left alone on purpose:** `FTSE100`,
  `AAPL130119` and `PAKISTAN92` stay because a digit touches the name; lower-case `fdic`, `nyse`, `aapl` stay (acronym case).
  The audit counts these as leftovers.
- **Result:** with the amendment, 94 of 500 dev questions contain an entity (it was 90 under the first build; demonyms add
  questions). With Britain added the audit leaves 0.45% (35 of 7,798, 25% level), 0.23% (57 of 24,056, 50%) and 0.30% (101 of 33,158, 100%) of occurrences, under the 2% rule; words that start with a name but are another
  thing (Indiana, the Americas) are counted apart (`NOT_THE_ENTITY` in `swap.py`).

**S4 (secondary, pre-registered now, before any M5 run).** Split the 94 affected questions by whether a **relevant passage
contains a swapped entity** (under the full alias set) or not: **72 questions have an entity in a relevant passage, 22 do not**
(of all 500 dev questions, 256 have one). The same tests as P1 to P3 on each group, reported with the group sizes first; the
group of 22 is small and its ranges will be wide. Reason: where a relevant passage names the entity, the swap changes both
sides and decides whether exact overlap survives (condition A) or breaks (condition B); where it does not, the swap only removes
the entity from the question, and no exact-match term between question and passage is lost. Expectation: A4's fall and the closing
gap to A4 are concentrated in the 72 and small in the 22. Secondary, none of it decides the verdict.

**Known residue (accepted on review, 2026-10-29).** Top 20 leftovers at the 100% level: fdic 16, PAKISTAN 8, Indianapolis 8,
nyse 8, FTSE 7, aapl 6, britains 4, AmericanExpress 3, tfsa 3, americancentury 3, rrsp 3, AAPL 2, americansecurityproject 2,
americanaffairsjournal 2, indianapolis 2, indianartvilla 2, canadataxplan 2, AmericanAir 2, russell 1, Pakistanee 1. They are
lower-case acronyms, names glued to digits or letters (PAKISTAN92, FTSE100, AmericanExpress) and other words that start with a
name (Indianapolis). Documents changed at 100%: 13,510.
