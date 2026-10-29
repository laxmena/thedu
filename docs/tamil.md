Status: parked. Draft, not a final pre-registration. Open decision: max length 8,192 (primary) vs 512 for the embedding model and reranker; Qwen3 sees about a quarter of a median passage at 512.

# Tamil (IndicQARetrieval): setup and pre-registration

Status: setup only. No retrieval, embedding, rerank or LLM run has been made on this dataset. This page is committed
before any number exists, which is what pre-registers the expectations below. Approved by the human on 2026-10-28
(a new dataset needs a human decision, AGENTS.md).

## Data

- Source: `mteb/IndicQARetrieval`, configs `ta-corpus`, `ta-queries`, `ta-qrels`, split `test`, license CC0
  (`cc0-1.0`). Pinned to revision `51e8b328988795d658f6f34acd34044e9346e2ee`, the revision MTEB evaluates
  (`mteb/tasks/retrieval/multilingual/indic_qa_retrieval.py`, read 2026-10-28). The files and revision are in
  `data/indicqa_ta/` (`REVISION.json`); `thedu data-fetch` rebuilds them (needs `uv run --with pyarrow`, which is not a
  project dependency: it is used once to turn parquet into jsonl).
- Counts match the expected ones: 253 passages, 1,803 questions, 1,804 judgments. One question
  (`c457553e...`, "அன்னை தெரசா எப்போது இறந்தார்?") has two relevant passages.
- Text is NFC-normalised when read (`thedu.data.loader`). In this dataset NFC changes nothing (0 texts changed).
  Zero-width joiners and non-joiners are kept as they are: 7 characters (2 ZWJ, 5 ZWNJ) in 4 passages, none in questions.
- Stats: `runs/analysis/indicqa_ta-stats.json` (`thedu data-stats`).

| | |
| --- | --- |
| Questions per passage | min 2, median 7, max 10 |
| Passage length, characters | min 769, median 1,854, max 6,940 |
| Question length, characters | min 19, median 50, max 196 |
| Passage tokens, bge-reranker-v2-m3 tokenizer | median 520; **52.6% (133 of 253) are over 512** |
| Passage tokens, Qwen3-Embedding tokenizer | median 2,029, min 851, max 7,603; **100% are over 512** |
| Questions whose relevant passage is missing | none |

**The 512-token limit is severe here.** Tamil is token-heavy for the Qwen tokenizer: at 512 tokens A4 sees about a
quarter of a median passage (the cross-encoder, query plus passage in 512, sees about 90% of a median one). BM25 sees
all of it. English used 512 with passages that mostly fit. I followed the instruction (512, as before) and flag the
consequence: any A4 or A4+R win over BM25 may partly be hidden by truncation, and any BM25 win may partly be caused
by it. Proposed, not decided: add a secondary arm "A4 at 8192 tokens" (the longest passage is 7,603 tokens; 253 short
documents, a few minutes to embed) to separate the two. MTEB's own published runs use the model's 32,768 limit.

## Split (by passage)

`configs/indicqa_ta_split.json`, frozen with `thedu freeze --add` (also the derived `data/qrels/indicqa_ta/{dev,test}.tsv`).
Rule: passage ids sorted, `random.Random(7).shuffle`, first floor(253/2) = 126 to dev, 127 to test; every question goes
with its relevant passage. Result: dev 126 passages, 885 questions; test 127 passages, 917 questions.

Two decisions you can reverse:

1. **One question is in neither split.** Its two relevant passages fell on different sides (one dev, one test), so
   it cannot go with both. Dropping it keeps the dev judgments free of test passages. 885 + 917 + 1 = 1,803.
2. **The corpus is shared**, all 253 passages in both splits, as in BEIR. Only questions and judgments are divided. A
   dev-only corpus of 126 passages would be easier than the MTEB task; with the full corpus the test score is
   comparable with MTEB's. Cost: a dev run also sees test passages as distractors (never their questions or answers).

Test questions load only through `thedu test-eval` (`data.load("indicqa_ta", "test")` raises, tested).

## Keyword settings (A0)

bm25s, PyStemmer `tamil` (Snowball), k1 = 0.9, b = 0.4, **no stopword list**; `bm25.language: tamil` in a config
(left out of the hash for English, so old run hashes are unchanged). Token pattern: Python's default `\b\w\w+\b` cuts
Tamil words at every vowel sign and virama (they are marks, not `\w`), so tokens are runs of `\w`, the Indic blocks
U+0900–U+0DFF (Hindi and Sanskrit words occur inside Tamil text) and the joiners. Single-character tokens are kept.

Stemmer, observed (`tests/test_tamil.py` prints and pins 12 words):

- It works on case endings: வீடு, வீட்டில், வீட்டுக்கு, வீடுகள் all become வீடு; புத்தகங்கள் / புத்தகத்தை become புத்தகம்;
  தமிழ்நாட்டின் / தமிழ்நாட்டில் become தமிழ்நாடு.
- It under-strips (பள்ளிகளுக்கு → பள்ளிகள், not பள்ளி) and over-strips: மாணவர்கள் → மாண, இந்தியாவில் → தியா (it drops a
  leading letter), என்பது → என்ப, கடவுள் → கட.
- It reduces some function words to the empty string (பற்றி "about", விட்டு, பட்டு, கொண்டிருக்கிறது): 210 of 74,955
  tokens (0.28%) in passages and questions. They stay in the index as one "" term; not removed, because the rule is
  "no stopword list". Worth a look before the first run.

## Embedding setup (A4)

Qwen3-Embedding-0.6B, query prefix `Instruct: <instruction>\nQuery:`, no instruction on documents, max sequence length 512.

- **Instruction: "Retrieve text based on user query."** IndicQARetrieval defines no `prompt` of its own, so MTEB falls
  back to the retrieval task class default, `AbsTaskRetrieval.abstask_prompt` (`mteb/abstasks/retrieval.py` line 97,
  mteb main, read 2026-10-28), via `get_instruction` in `mteb/models/abs_encoder.py`. The Qwen3 template is in
  `mteb/models/model_implementations/qwen3_models.py`. Caveat: this is mteb main today; the published Qwen3 results for
  the other tasks were made with mteb 1.38.9, and I did not verify that its fallback was the same text. It is in
  `TASK_INSTRUCTIONS` as `indicqa_ta`.
- **Published score: none found.** The results repo (`embeddings-benchmark/results`, folder
  `Qwen__Qwen3-Embedding-0.6B/b22da495...`) has no `IndicQARetrieval.json` (checked by listing the folder through the
  GitHub API, 2026-10-28). I found no published BM25 score for the Tamil subset either (searched the web; a Hindi-BEIR
  paper lists IndicQARetrieval for Hindi only). Nothing is pinned in `tests/expected.yaml` (it is frozen and I added
  no line), so there is **no external parity check for Tamil**. The check that remains is internal: A4 above A0 would
  be reassuring, and any A4 number is reported without a published reference.

## Pre-registration

Written before any number exists. Dev first (`n` = all 885 dev questions), then test once, through `thedu test-eval`.

**Arms**

| Arm | What it is |
| --- | --- |
| A0 | BM25, Tamil stemmer, no stopwords |
| A3 | A0 with the grounded rewrite: top 5 shown to the model, question repeated 3 times plus the added terms |
| A0+R, A3+R | top 50 of A0 / A3 reordered by bge-reranker-v2-m3, zero-shot, max length 512, rest kept in order |
| A4 | Qwen3-Embedding-0.6B, exact cosine, 512 tokens, instruction above |
| A4+R | A4 top 50 through the same cross-encoder |

**Rewrite prompt** (`src/thedu/prompts/grounded_ta_v1.txt`, not yet wired into the config): the same shape as
`grounded_v1`, adapted: the model is told the passages are Tamil, to output up to 15 terms in Tamil script, to keep the
question's words, to add other word forms of its key words (endings, plurals, case suffixes) and Tamil synonyms, and to
keep Latin-script names exactly as they appear. The question is kept as in English (repeated 3 times). The rewriter is the
pinned qwen/qwen3-14b on DeepInfra; I have not checked how well it writes Tamil, and the first look at its output will
be on dev, not on a number.

**Primary tests** (paired, 95% bootstrap range, same wide-range rule as before: a miss counts as "the other is better"
only if the upper bound is below 0, otherwise inconclusive):

- **T1: A3 vs A0**, nDCG@10. "A3 beats A0" if the lower bound is above 0.
- **T2: A3+R vs A4+R**, nDCG@10. "Match" if the lower bound of (A3+R − A4+R) is above −0.01; "beat" if above 0.

Dev first; then test once, with the same two tests and nothing changed after seeing dev. Secondary (reported as such,
nothing decided by them): A0+R vs A4+R, A3+R vs A4, A0 vs A4, A4+R vs A4, Recall@50 and Recall@100 for A0, A3, A4,
cost per search, and A4 at 8192 tokens if you approve it.

**Expectations (written before any number)**

1. **T1 holds on dev: A3 beats A0**, with a smaller gain than in English (those were +0.02 to +0.06 on three
   datasets). Reasons for it: Tamil nouns carry many endings and the stemmer is imperfect, so extra word forms help;
   the passages are long, so the first-round top 5 is rich in the passages' own wording. Reasons against: a 14B model
   may write poor Tamil, and a wrong form adds noise. If the rewrite model writes mostly non-Tamil output the test
   fails and that is the finding.
2. **T2 does not hold; A4+R is better than A3+R.** Two reasons: BM25 candidates miss where the question and passage use
   different wording (the English reranker runs showed A3R below A4 on all three English sets), and the cross-encoder
   is multilingual (bge-reranker-v2-m3 was trained on Tamil data) while the keyword side is not. This is the opposite of
   the project hope, stated plainly before the result.
3. Level guesses, so surprises are visible: A0 nDCG@10 between 0.45 and 0.75 (253 passages is a small, easy corpus);
   A4 above A0; A4+R at or above A4. If A4 (512 tokens) lands below A0, I treat truncation as the first suspect, not
   the model.
4. Mechanism notes: the cross-encoder sees query plus passage in 512 tokens, so 47% of the passages (133 of 253) are
   cut for it too; the relevant sentence is often later in the passage (cloze-style questions), so reranking top 50 can
   lose recall to truncation. A0 and A3 are not affected by length.

**Tanglish plan.** After the dev and test runs, you retype about 100 test questions in Latin script (Tanglish), keeping
the same judgments (same question ids, same relevant passages). I run A0, A3, A0+R, A3+R, A4, A4+R on those 100 and on
the same 100 in Tamil script (paired, same ids), for one test-split comparison, not tuned. Expectation: A0 collapses
(passages are in Tamil script, so almost no tokens match), A3 recovers part of it if the rewrite model transliterates
back into Tamil script, and A4 and the reranker fall less. How the 100 are chosen (a seeded sample of test question ids
from the split file) and who types them is yours to settle; the questions are test questions, so the retyped file is
treated as test data and read only by `thedu test-eval`.

## Not done yet (waits for your review)

Wiring the Tamil prompt into the planner config (`PlannerConfig.grounded_prompt` allows only `grounded_v1` today), the
dataset configs under `configs/`, `thedu parity` / `test-eval` support for Tamil, the 8192-token A4 arm, and all runs.
