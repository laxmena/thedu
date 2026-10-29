# Journal

Decisions, attempts, findings, learnings, blockers and observations, in order.
Append only. See AGENTS.md for the format.

## 2026-10-27 · observation · The starting question
Who: human
What: A search product claiming semantic search without embeddings made me curious whether you can skip embeddings and put the understanding into the query instead.
Why it matters: It's the question the whole project answers.
Evidence: none yet
Next: Build my own version and measure it against embeddings.
Post: yes

## 2026-10-27 · decision · A blog post with a research repo, not a product
Who: human
What: Build this as an INTERNALS.md post with a reproducible repo. No library, no service, no support promises.
Why it matters: Keeps the scope to a few weekends and the focus on findings.
Evidence: none
Next: Label the repo as research code in the README.
Post: no

## 2026-10-27 · decision · Public datasets only
Who: human
What: Use only public data (SciFact, NFCorpus, FiQA, BRIGHT, Project Madurai) plus Tamil questions I write myself. No personal notes, no work data.
Why it matters: Anyone can reproduce every number, and nothing private leaks.
Evidence: none
Next: Start with SciFact and NFCorpus, which run on a laptop.
Post: no

## 2026-10-27 · decision · Pyserini as the reference, bm25s as the fast engine, not SQLite FTS5
Who: human
What: Pyserini proves correctness against published baselines; bm25s runs inside the loop. SQLite FTS5 is out for the study because its BM25 can't match published scores.
Why it matters: If baselines don't match published numbers, no later result can be trusted.
Evidence: none yet; parity tests in M1 will confirm
Next: Pin published scores in tests/expected.yaml.
Post: maybe, as a short "drop it into your app" demo with FTS5

## 2026-10-27 · decision · Prompt only in v1; RL later; no supervised fine-tuning
Who: human
What: v1 uses Qwen3-4B as is and improves only the prompt. v2 may train a small model with RL against BM25 results.
Why it matters: There are no correct rewrites to copy, only a score for how well a rewrite retrieves, which is what RL learns from.
Evidence: none
Next: Revisit after M5 shows where the prompted model falls short.
Post: yes, as the part 2 teaser

## 2026-10-27 · decision · Start with bm25s alone; Pyserini becomes a fallback
Who: human
What: Revises the earlier keyword-search decision. bm25s is the only engine at the start, set up like Anserini's flat baseline. Pyserini comes in only if a trigger fires (see System design, Fallbacks).
Why it matters: Removes Java from the setup. The parity test against Anserini's published flat BM25 scores still proves correctness.
Evidence: none yet; M1 parity will confirm or trigger the fallback
Next: Pin Anserini flat (A0) and MTEB (A4) reference scores in tests/expected.yaml.
Post: maybe, if parity turns out to be harder than expected

## 2026-10-27 · decision · One small Python CLI for every command
Who: human
What: All commands run through `uv run qtr <command>` instead of a Makefile. qtr = query-time retrieval, a placeholder name.
Why it matters: Humans and agents run the exact same commands, and the gate can't be bypassed by accident.
Evidence: none
Next: Rename together with the post title.
Post: no

## 2026-10-27 · decision · The CLI is called thedu
Who: human
What: Renamed the CLI from qtr to thedu (தேடு, Tamil for "search!"). Commands are now `uv run thedu <command>`. Thedal (the noun, "search") was the runner-up.
Why it matters: A name with meaning, and it ties the tool to the Tamil track.
Evidence: none
Next: Post title still open.
Post: maybe, as a one-line aside on the name

## 2026-10-27 · decision · Search is the default command
Who: human
What: `thedu "question"` searches, showing each round's rewrite and the top results with sources. `--compare` shows embedding results side by side. Research commands (check, dev, gate) stay as subcommands.
Why it matters: The name reads as a sentence ("search: question"), and readers can feel the difference in one command.
Evidence: none
Next: Build in M2, alongside the planner.
Post: yes, as the opening "try it yourself" line

## 2026-10-27 · learning · A critical review of the plan found three real problems
Who: human + Claude
What: (1) Four milestones of infrastructure came before any signal, so a one-evening spike now comes first. (2) Baseline parity needs test answers, which the leakage guard blocked, so `thedu parity` is now the single explicit exception. (3) The gate compared overlapping confidence ranges, which would reject most real 1–2 point gains, so it now uses a paired test question by question.
Why it matters: Each would have cost weeks or quietly made the results less trustworthy.
Evidence: plan review, no runs yet
Next: Also fixed: cost priced at reference hosted rates (local isn't free), BRIGHT is evaluation only and moved earlier to M2, the made-up names test gets a design page, the sandbox and spend caps move to M0, and M7 is cut to two headline exhibits.
Post: yes, a good "plans are hypotheses too" aside

## 2026-10-27 · blocker · The spike's default model name didn't exist
Who: human + Claude
What: The first run used `qwen/qwen3-8b`, which OpenRouter doesn't list. All 100 rewrites failed with NotFoundError after three retries each, and the script would have scored "A2" with empty rewrites, which is just BM25 again.
Why it matters: A silent failure like this produces a believable but meaningless number.
Evidence: first spike run, console output
Next: The script now checks the model and key once before starting, and refuses to report A2 if more than 5% of rewrites fail. Default model is now `qwen/qwen3-14b`.
Post: maybe, as a one-line "fail loudly" lesson

## 2026-10-27 · finding · First spike: one rewrite doesn't help BM25 on SciFact; embeddings lead
Who: human + Claude
What: 100 SciFact dev questions, rewriter qwen/qwen3-14b, prompt v1. nDCG@10: BM25 0.701, embeddings 0.757, rewrite + BM25 0.695. Rewrite vs BM25: −0.007 (95% range −0.060 to +0.046), 18 wins, 23 losses. Embeddings vs BM25: +0.056 (range −0.005 to +0.120), 28 wins, 15 losses.
Why it matters: The simplest form of the idea shows no gain. Embeddings look ahead but the range still touches zero at 100 questions.
Evidence: runs/spike/history/20261027-192518-qwen3-14b-v1-q1-n100.json
Next: Reading the rewrites explains a lot: they average 42 words (the prompt asked for at most 30, one ran to 213), repeat words ("sepsis" over a dozen times), and pad with generic research words ("clinical trial", "research methodology", "immune response"). Those drown out the original claim. Try one change at a time: prompt v2 (≤12 specific new terms, no generic words, no repeats), then repeating the original question so it outweighs the additions. Also, SciFact claims already use the abstracts' technical vocabulary, so the word gap rewriting fixes may be small here; BRIGHT and NFCorpus may tell a different story.
Post: yes, the honest first result

## 2026-10-27 · decision · Keep every spike run instead of overwriting
Who: human + Claude
What: Each run now also saves runs/spike/history/<time>-<model>-<prompt>-q<repeat>-n<n>.json; summary.json is just the latest. The first run's summary was copied into history so nothing is lost.
Why it matters: The trial-and-error record is the raw material for the post.
Evidence: scripts/spike.py
Next: The same rule applies to the real run registry in M0.
Post: no

## 2026-10-27 · finding · Why blind rewriting backfires on SciFact: dilution and drift
Who: human + Claude
What: Replayed all 100 questions with the exact cached rewrites (replay matches the runs to four decimals). Three causes stack up. (1) Dilution: BM25 sums every query word and bm25s counts repeats, so added words take weight from the claim. The claim's own words carried 52% of query weight with v1 and only 36% with v2. (2) Drift: 63% of the claim's words appear in the correct abstract, but only 38% of v1's added words and 28% of v2's. The added words describe the claim's neighbourhood, and the neighbourhood is full of other abstracts. (3) Little to fix: BM25 already had a correct #1 for 57 of 100 questions and missed entirely on only 13; blind rewriting rescued 3 of those 13 and pushed down 11 (v1) or 15 (v2) of the 57.
Why it matters: It explains the result and points to the fix: keep the question in charge and ground the added words in the collection.
Evidence: replay of runs/spike/history/*v1-q1* and the v2 run; example: "Pyridostatin induces double-strand breaks accumulation" lost its correct #1 to a general DNA-repair review after v2 added "DNA, damage, pathway, repair, homologous recombination".
Next: Repeating the question recovers it: v1 ×2 +0.027, v1 ×3 +0.029 (range −0.014 to +0.074), v2 ×3 +0.005. Promising, not proven at 100 questions.
Post: yes, this is the core mechanism section

## 2026-10-27 · learning · What the query-expansion papers already say (from memory, verify before citing)
Who: Claude
What: query2doc repeats the original query several times (about five) when adding LLM text for BM25, which matches our dilution finding. Corpus-steered expansion (CSQE) and LameR ground the expansion in documents the search actually returned, which addresses drift. ThinkQE reasons first and refines over rounds with retrieved documents. Jagerman et al. found smaller models benefit most from seeing retrieved documents.
Why it matters: Our blind A2 skipped both safeguards; A3 is designed to include them.
Evidence: none yet; citations to check in M7
Next: A3 uses the question ×3 and the top-5 results as context.
Post: yes, with proper citations

## 2026-10-27 · decision · Stop tuning prompts on these 100 questions; finish the spike with A3, NFCorpus and one larger run
Who: human + Claude
What: Six variants have now been tried on the same 100 SciFact questions, so more prompt tuning there risks fitting luck. Last spike steps: (1) A3, grounded rewriting: BM25 first, the LLM reads the top 5 and writes terms from the relevant ones, search again with the question ×3; (2) the same setups on NFCorpus dev, where questions and documents use different words; (3) one run on all 809 SciFact dev questions. The spike's defaults changed to match: question ×3 for A2 and A3; v1 stays the A2 prompt.
Why it matters: Tests whether the problem is rewriting itself or rewriting without context, and whether SciFact simply has too little word gap.
Evidence: scripts/spike.py
Next: If A3 beats BM25 anywhere, close the spike and start M0. If nowhere, the post becomes "why rewriting without the right context backfires" and we discuss before building M0–M4.
Post: yes

## 2026-10-27 · finding · Grounded rewriting (A3) stops the damage but doesn't add much on SciFact
Who: human + Claude
What: 100 SciFact dev questions, question ×3, one feedback round. nDCG@10: BM25 0.701, A3 0.703, A2 (v1 ×3) 0.730, embeddings 0.757. A3 vs BM25 +0.002 (range −0.030 to +0.035); A3 vs embeddings −0.054 (range −0.118 to +0.005). A3 pushed down only 5 of BM25's 57 correct #1s (blind v1 ×1 pushed down 11, v2 15) but rescued only 3 of the 13 misses.
Why it matters: Reading the top 5 cured most of the drift, but it can only reuse words from results BM25 already found. When the top 5 are wrong it copies their words (e.g. "lumbar intervertebral disc herniation" for a question on integrated care). This is the known limit of pseudo-relevance feedback.
Evidence: runs/spike/history/20261027-202031-scifact-A0-A4-A2-A3-qwen3-14b-v1-q3-r1-n100.json
Next: A3 and embeddings win on different questions: picking the better of the two per question would score 0.83, and A3 beats embeddings on 18 questions. That's a strong hint for a hybrid (A5 in M3). Also: the v2 ×2 run scored 0.6784, exactly what the offline replay predicted, so the setup is reproducible. Every difference except v2 ×1 is still inside its range at 100 questions; NFCorpus and the 809-question run come next.
Post: yes

## 2026-10-27 · finding · NFCorpus: grounded rewriting gives the first real gain
Who: human + Claude
What: 100 NFCorpus dev questions, question ×3, one A3 round. nDCG@10: BM25 0.283, A2 blind 0.309, embeddings 0.318, A3 grounded 0.326. A3 vs BM25 +0.043 (range +0.017 to +0.071), the first comparison in the spike whose whole range is above zero. A3 vs embeddings +0.008 (range −0.037 to +0.056): a tie. A2 vs BM25 +0.026 (range −0.016 to +0.068).
Why it matters: Where questions are short and plain (median 2.5 words, e.g. "dulse", "Glycotoxins") and the documents are technical, rewriting closes the vocabulary gap, as predicted. BM25 found nothing correct for 34 of 100 questions here, against 13 of 100 on SciFact.
Evidence: runs/spike/history/20261027-203634-nfcorpus-A0-A4-A2-A3-qwen3-14b-v1-q3-r1-n100.json
Next: Best examples. A3 turned "Glycotoxins" into "advanced glycation endproducts, AGEs, methylglyoxal" (0.00 → 0.47). The query "California Raison Marketing Board" is misspelled; A3 picked up "raisin" from the results and went 0.00 → 0.69. Blind A2 turned "dulse" into "seaweed, red algae" (0.00 → 0.63). The losses are drift from wrong top results again: "phenolics" picked up "Salmonella typhimurium mutagenicity" and fell 0.36 → 0.14. A3 and embeddings again win on different questions: the better of the two per question would score 0.40, which supports a hybrid. Caveats: one significant result among about ten comparisons can be luck, so confirm on all 324 NFCorpus dev questions. Our BM25 (0.283 on dev) is below the published ~0.32 on test; that's for M1 parity to check.
Post: yes, this is the headline result so far

## 2026-10-27 · finding · Full runs confirm the pattern: rewriting helps only where the words differ; embeddings win alone
Who: human + Claude
What: All dev questions, question ×3, one A3 round. NFCorpus (324): BM25 0.285, A2 0.314, A3 0.325, embeddings 0.336. A3 vs BM25 +0.040 (range +0.024 to +0.055); on the 224 questions not in the first run alone, +0.039 (range +0.021 to +0.058), so it replicates. A3 vs embeddings −0.011 (range −0.034 to +0.011), a tie. SciFact (809): BM25 0.695, A3 0.687, embeddings 0.738. A3 vs BM25 −0.007 (range −0.018 to +0.003), no gain. Embeddings vs BM25 +0.043 (range +0.023 to +0.064). A3 vs embeddings −0.050 (range −0.072 to −0.028).
Why it matters: The spike's question is answered. Grounded rewriting closes a real vocabulary gap (NFCorpus) and does nothing where there isn't one (SciFact). Embeddings beat BM25 on both and beat rewriting on SciFact. On its own, query-time rewriting is not a replacement for embeddings.
Evidence: runs/spike/history/20261027-204338-nfcorpus-…-n324.json and 20261027-204716-scifact-A0-A4-A3-…-n809.json
Next: The strongest signal is that the two win on different questions. Picking the better of A3 or embeddings per question would score 0.805 on SciFact (vs 0.738) and 0.395 on NFCorpus (vs 0.336). A hybrid (A5 in M3) is where the gain probably is, and the post's question shifts from "replace embeddings?" to "what does query-time rewriting add on top of them, and at what cost?" Cost so far: about 610 tokens per question for A3 on SciFact (494K for 809).
Post: yes, this is the spine of the post

## 2026-10-27 · decision · Close the spike and start M0
Who: human + Claude
What: The spike met its stopping rule ("A3 helps on NFCorpus but not SciFact: start M0"). Lessons carried into M0: question ×3 by default; grounding in retrieved documents; write rewrites and rankings to disk as they arrive, not at the end; save top-10 rankings in TREC run format; more parallel LLM calls; check the model before a run; keep every run.
Why it matters: M0 builds on measured behaviour, not on the original guess.
Evidence: this journal, 2026-10-27 entries
Next: M0 in a Claude Code session working from AGENTS.md. Hybrid (A5) moves up in priority.
Post: no

## 2026-10-27 · decision · M0 layout: src/thedu, plain-text prompts behind a Plan wrapper, and three guards
Who: human + Claude
What: Code lives in `src/thedu/` (judge, data, retrievers, planner, ranker, prompts as subpackages); AGENTS.md now names the frozen paths that way. Prompts stay plain text (the LLM returns search terms, not JSON) so the spike's cached rewrites keep working; the planner wraps terms in a validated `Plan` and falls back to the raw question after one retry. Test answers sit behind one function that only `thedu parity` and `thedu test-eval` import, enforced by a grep test instead of a call-stack check. A run records its git commit plus a hash of the package source and refuses to resume under different code.
Why it matters: Keeps the spike's numbers reproducible while M0 builds the real parts, and makes leakage and mixed-code runs hard to do by accident.
Evidence: AGENTS.md, src/thedu/, tests/test_data.py, tests/test_registry.py
Next: JSON-output prompts arrive with the multi-round planner (M2). The code hash is a small addition to "record the commit": with an uncommitted tree, the commit alone could not tell two code versions apart.
Post: no

## 2026-10-27 · learning · M0 done: the spike's SciFact numbers reproduce from the new package
Who: human + Claude
What: `thedu dev` on the spike's settings, with cached rewrites and no network, gives A0 0.6945, A3 0.6874 and (embeddings) A4 0.7376 on all 809 SciFact dev questions, matching the spike. The judge matches pytrec_eval to 1e-12 on stored BM25 runs for SciFact and NFCorpus (graded relevance). Dataset sizes match the published counts for SciFact (5183 docs, 809 dev, 300 test), NFCorpus (3633, 324, 323) and FiQA (57638, 500, 648); FiQA is downloaded but not embedded. Top-10 of BM25 is identical at k=10 and k=100 on 100 questions, so keeping 100 results per question (for Recall@100) did not change nDCG@10. `thedu check` takes about 3 seconds; `thedu smoke` about 30.
Why it matters: The foundation is trustworthy before any new method is built on it.
Evidence: tests/test_reproduction.py, tests/smoke/test_smoke.py, tests/golden/. Recall@100: A0 0.926, A3 0.941, A4 0.959; A4 vs A0 +0.043 nDCG@10 (range +0.022 to +0.065), A3 vs A0 -0.007 (range -0.018 to +0.005).
Next: Two known gaps. (1) The spike's cached rewrites recorded no token split, so A3's cost per search shows n/a until its rewrites are re-run through the gateway (new calls are priced at $0.12 / $0.24 per million tokens, qwen/qwen3-14b on OpenRouter, fetched 2026-10-27). (2) Embedding search has no reference price yet, so its cost shows n/a. Then M1: parity against published scores.
Post: no

## 2026-10-27 · decision · Published reference scores frozen; `thedu freeze --add` appends only
Who: human + Claude
What: `tests/expected.yaml` holds the published test nDCG@10 that parity checks against, each with source URL, fetch date and the setup the source describes. Anserini BEIR v1.0.0 flat BM25: SciFact 0.6789, NFCorpus 0.3218 (docs page in the Anserini repo). MTEB results for Qwen3-Embedding-0.6B (mteb 1.38.9, model revision b22da49): SciFact 0.69718, NFCorpus 0.36709. Tolerance is an absolute difference of 0.010. The file was added to `tests/frozen.sha256` with `thedu freeze --add tests/expected.yaml`. That command only appends lines for files named explicitly and never changes or removes an existing line; this is the first addition after the M0 freeze.
Why it matters: The numbers we must match cannot be adjusted after seeing our own results.
Evidence: tests/expected.yaml, tests/frozen.sha256, tests/test_frozen.py, tests/test_expected.py
Next: Any later file frozen this way gets its own journal line. The rendered Anserini page on castorini.github.io returned 404 when fetched, so the value was read from the page's source in the GitHub repo, which says it is the page's own generated table.
Post: no

## 2026-10-27 · finding · M1 parity passes: A0 and A4 are within 0.006 of the published test scores
Who: human + Claude
What: Test-split nDCG@10, ours vs published. Attempt 0 (defaults: bm25s k1=0.9 b=0.4, embeddings at 512 tokens with the model's generic query prompt): A0 SciFact 0.6762 vs 0.6789 (-0.0027), A0 NFCorpus 0.3179 vs 0.3218 (-0.0039), A4 SciFact 0.6972 vs 0.6972 (+0.0000), A4 NFCorpus 0.3566 vs 0.3671 (-0.0105, just over the 0.010 limit), so parity failed on one number. Fix 1, the only fix tried: put MTEB's task instruction in front of each query ("Instruct: Given a question, retrieve relevant documents that best answer the question" for NFCorpus, the claim-checking one for SciFact). Result: A4 SciFact 0.7001 (+0.0029), A4 NFCorpus 0.3617 (-0.0054). All four within tolerance, parity passes.
Why it matters: The gate is open. The BM25 and embedding baselines reproduce published numbers, so later gains are measured against baselines that can be trusted.
Evidence: runs/55e0f4c1e7 and runs/47f7f0d24b (attempt 0); runs/fc4262ace8 and runs/1cb69a2a1e (fix 1); runs/c9b67a9ab1 and runs/850fbee2c4 (final confirmation after tidying the code, same numbers). Each has parity.json.
Next: What I learned. (1) BM25 needed no change: bm25s's "en" stopword list is exactly Lucene's 33-word English set, so the list I was told to check first is not a difference. Two differences remain and were not needed: Snowball vs Porter stemming, and bm25s's tokenizer vs Lucene's StandardTokenizer. Our A0 is a little below Anserini on both sets (-0.003, -0.004), which is inside tolerance, and I do not know why. (2) A4: MTEB runs with the model's full 32768-token limit; we use 512. 9.0% of SciFact documents and 9.6% of NFCorpus documents are longer than 512 tokens. Parity passes anyway, so the limit stays at 512 and nothing was re-embedded; this is a known difference, not a fixed one. (3) SciFact A4 matched MTEB to four decimals with the generic prompt, which is not the prompt MTEB's configuration says it used, so I treat that as chance and kept the documented instruction. (4) Test answers were read in three parity runs per dataset, to reproduce published numbers (one fix tried, no tuning for a higher score). (5) The task instruction is now the default for new configs (`query_instruction: task`); the spike configs and champion/challenger are pinned to `model-default`. On dev it moves A4 only slightly: SciFact 0.7376 to 0.7368, NFCorpus 0.3363 to 0.3373 (runs/0511fd41c5, runs/75abfee242, runs/6fd1962ecb). BM25 settings did not change, so A0 and A3 on dev did not need re-running. FiQA has no verified instruction yet, so A4 on FiQA stops with an error rather than guessing.
Post: yes, as the "we reproduced the published baselines first" section

## 2026-10-27 · observation · BM25 results depend on k when scores tie
Who: Claude
What: On NFCorpus dev, A0 scores 0.2854 nDCG@10 when 10 results are kept and 0.2838 when 100 are kept (same engine, same settings). 90 of 324 questions get a different top 10; 113 have tied scores inside the top 11 (many questions match nothing, so documents tie at zero). bm25s breaks ties in an order that changes with k. SciFact showed no difference on the 100 questions I checked in M0, and its pinned reproduction numbers (0.6945, 0.6874) are computed at k=100.
Why it matters: "Same config, same numbers" holds because k is fixed, but tied documents are ordered arbitrarily, so a 0.002 change can come from the tie order alone. This also corrects my M0 note that the top 10 was identical at k=10 and k=100: that held for the SciFact sample, not for NFCorpus.
Evidence: check run on NFCorpus dev, not saved as a run; reproduce with Bm25Retriever.search(q, 10) vs search(q, 100)
Next: Not fixed in M1, because ordering ties by document position would move the pinned numbers and need a parity rerun. Proposed: break ties by document order inside the retriever, then re-pin the reproduction numbers and rerun parity. Needs a human OK.
Post: maybe, as a short "ties" aside

## 2026-10-27 · decision · One tie rule for every retriever: score, then doc id (largest first), whole collection ranked
Who: human + Claude
What: Retrievers now rank the whole collection by score, break ties by document id with the larger id first (trec_eval's convention), and cut to k afterwards. Before, bm25s ordered tied documents in a way that changed with k (see the 2026-10-27 observation on ties). A test now checks that the top 10 equals the first 10 of a longer list across 150 NFCorpus dev questions plus empty and unmatched queries. The dense retriever uses the same rule. The reproduction pins in `thedu check` stay at the SciFact values and gain NFCorpus values (below); `thedu smoke` keeps A4 at 0.7376. Parity was rerun once, as allowed after a retriever change.
Why it matters: "Same config, same numbers" no longer depends on how many results are kept, and our ranking order matches the one the judge's reference tool (trec_eval, pytrec_eval) would use for ties.
Evidence: src/thedu/retrievers/order.py, tests/test_retrievers.py, tests/test_reproduction.py; superseded "before" runs are kept as runs/*.superseded-*
Next: The spike's own figures (SciFact A0 0.6945, A3 0.6874; NFCorpus A0 0.285, A3 0.325) are historical: they were taken at k=10 with the old tie order. SciFact reproduces exactly under the new rule; NFCorpus differs in the third decimal.
Post: no

## 2026-10-27 · finding · The tie rule moved NFCorpus by 0.0004 nDCG@10 and nothing else that matters
Who: human + Claude
What: Dev, all questions, k=100, before (old tie order) vs after (new rule). SciFact (809): A0 0.6945 vs 0.6945, A3 0.6874 vs 0.6874, Recall@100 0.9264 / 0.9406 unchanged; A3 vs A0 -0.0070 (range -0.0181 to +0.0048) both times. NFCorpus (324): A0 0.2838 vs 0.2842, A3 0.3219 vs 0.3221; A3 vs A0 +0.0381 (range +0.0234 to +0.0529) vs +0.0379 (range +0.0236 to +0.0523). NFCorpus Recall@100 for A0 fell from 0.2376 to 0.2269 and for A3 from 0.3091 to 0.3085. Parity on test: SciFact unchanged; NFCorpus A0 0.3179 to 0.3177 (diff -0.0041), A4 0.3617 to 0.3613 (diff -0.0058); still PASS.
Why it matters: The ordering fix does not change any conclusion. The larger drop in NFCorpus A0 Recall@100 is a reminder that many NFCorpus questions match fewer than 100 documents, so the rest of the top 100 is filler with score zero; which filler is chosen is arbitrary (now: largest ids) and can move Recall@100 by about a point without meaning anything. Treat NFCorpus Recall@100 as soft.
Evidence: before: runs/566043e182.superseded-* and runs/e071f828ec.superseded-* (dev), runs/c9b67a9ab1.superseded-* and runs/850fbee2c4.superseded-* (parity); after: runs/566043e182, runs/e071f828ec, runs/c9b67a9ab1, runs/850fbee2c4. Running NFCorpus A3 again cost 63 new rewrite calls because its first-search top 5 changed for some questions (the rewrite cache is keyed on those five documents); the cache in runs/cache/llm.jsonl now covers both datasets, so `thedu check` stays offline.
Next: Pinned in check: NFCorpus A0 0.2842, A3 0.3221. Read-only parity reads were not repeated beyond the one rerun.
Post: no

## 2026-10-27 · decision · No per-run NOTES.md; the run folder is the record
Who: human
What: Removed the rule that every run writes `runs/<hash>/NOTES.md` in five lines. The run folder (config, per-question scores, rankings, metrics) records what ran and what it scored; JOURNAL.md holds the reasoning. AGENTS.md ("After every run") and the `thedu dev` closing message were updated.
Why it matters: One place for reasoning instead of two, and no extra step on every run.
Evidence: AGENTS.md, src/thedu/cli.py
Next: Journal entries link to run folders when a run taught something.
Post: no

## 2026-10-27 · finding · M3: the rewrite adds nothing on top of embeddings on SciFact, and a small real gain on NFCorpus
Who: human + Claude
What: Dev, all questions, nothing tuned (RRF k=60, equal weights, top 100 of each list, zero-score BM25 results dropped before fusion). H0 = RRF(BM25, embeddings), A5 = RRF(grounded rewrite, embeddings). nDCG@10 (Recall@100): SciFact (809) A0 0.6945 (0.926), A3 0.6874 (0.941), A4 0.7368 (0.954), H0 0.7409 (0.970), A5 0.7357 (0.974). NFCorpus (324) A0 0.2842 (0.227, soft), A3 0.3221 (0.309, soft), A4 0.3369 (0.329), H0 0.3364 (0.330), A5 0.3514 (0.363). The key comparison, A5 vs H0 (paired, 95% range): SciFact -0.0052 (-0.0134 to +0.0032), wins 76, losses 97, so A5 does not beat H0; NFCorpus +0.0150 (+0.0039 to +0.0253), wins 123, losses 81, so A5 beats H0, with the low end of the range close to zero. Other pairs. A5 vs A4: SciFact -0.0011 (-0.0175 to +0.0140), NFCorpus +0.0145 (-0.0026 to +0.0302). H0 vs A4: SciFact +0.0041 (-0.0114 to +0.0195), NFCorpus -0.0005 (-0.0141 to +0.0116), so plain hybrid fusion alone is no better than embeddings here. A5 vs A3: SciFact +0.0483 (+0.0349 to +0.0605), NFCorpus +0.0293 (+0.0172 to +0.0429). Recall@100, A5 vs H0: SciFact +0.0043 (-0.0019 to +0.0111), NFCorpus +0.0333 (+0.0218 to +0.0469).
Why it matters: Where the question and the documents use different words (NFCorpus), the rewrite brings in documents embeddings miss, and fusing it with embeddings is better than either alone for recall (+3.5 points over A4, range +2.2 to +5.0). Where they already share words (SciFact), it adds nothing over plain fusion; the SciFact recall gain over A4 (+0.020, range +0.009 to +0.032) is real but does not reach the top 10.
Evidence: runs/2a4eb7b8d0 (SciFact), runs/f8a880adfa (NFCorpus); configs/m3_scifact.yaml, configs/m3_nfcorpus.yaml. Rescued misses (A4 found nothing relevant in its top 10): SciFact A5 rescues 32 of 119 (H0 also 32) and pushes down 67 of 484 A4 correct-at-#1 (H0 58); NFCorpus A5 rescues 15 of 89 (H0 14) and pushes down 5 of 12 (H0 6). Best of A3 or A4 per question, the ceiling: SciFact 0.8031, NFCorpus 0.3941, against A5's 0.7357 and 0.3514; fusion by rank recovers none of the 6.6-point gap between A4 and the ceiling on SciFact (A5 is 0.1 point below A4; H0 recovers 0.4) and about a quarter of the 5.7-point gap on NFCorpus (A5 is 1.45 points above A4). Cost per search for the rewrite, priced at $0.12 / $0.24 per million tokens: SciFact about $0.00009 (estimated), NFCorpus about $0.00008 (estimated); H0 has no LLM cost. Embedding search itself is not priced yet, so total cost per search is n/a for both. Latency, relative to BM25 alone, with the rewrite call timed from 30 fresh calls (median 1.2 s per call, both datasets): A5 about 1970x (SciFact) and 4300x (NFCorpus) in parallel (max of A3 and A4), 1-2% more if run one after the other; H0 about 33x and 39x. So A5 is roughly 60x (SciFact) and 110x (NFCorpus) the latency of H0. These are relative figures only; the LLM call dominates.
Next: Cautions. Eight paired comparisons were reported across two datasets, and only A5 vs H0 on NFCorpus (the pre-named key comparison) clears zero with a low end of +0.004; treat it as promising, not proven, until replicated on another dataset (FiQA is the next candidate once it has a verified embedding instruction). Nothing here was tuned, so no tuning luck is involved, but RRF weights and k, and a better planner, are what M4 will try. The cross-encoder rerank is the optional next step; the number to beat is H0 and A5 at 0.7409/0.7357 (SciFact) and 0.3364/0.3514 (NFCorpus).
Post: yes, this is the central result: rewriting adds value only where there is a word gap, and what it adds is mostly recall

## 2026-10-27 · learning · Cost can be estimated offline from cached rewrites; the spike's "494K tokens for 809 questions" covered only 709 calls
Who: Claude
What: To price the spike's cached rewrites (no input/output split was recorded), prompts are counted with the Qwen3 tokenizer plus the chat template, and answers as cached text plus the end-of-turn token (src/thedu/tokens.py). For the 709 SciFact A3 calls that the spike billed, the estimate is 492,273 tokens against 493,749 recorded: 0.30% low, so hidden thinking tokens are a rounding error here. For all 809 calls the estimate is 560,975 (620.9 input and 73.4 output tokens per call). The earlier journal line "about 610 tokens per question for A3 on SciFact (494K for 809)" was wrong: the spike's counter only counts calls that were not already cached, and the first 100 questions came from the earlier 100-question run, so the real figure is about 696 tokens per call.
Why it matters: Costs shown with a "~" are estimates good to well under 1% on this evidence. It also means the spike's other token totals count new calls only.
Evidence: runs/2a4eb7b8d0 (per-call tokens in per_question.jsonl), runs/spike/history/20261027-204716-scifact-A0-A4-A3-qwen3-14b-v1-q3-r1-n809.json (493749 tokens)
Next: Fresh calls are priced from the provider's own token counts, so estimates disappear as the cache is regenerated.
Post: no

## 2026-10-27 · decision · The ranker drops zero-score BM25 results before fusion; only fusion inputs, not A0 or A3 results
Who: human + Claude
What: `thedu.pipeline.fuse` removes zero-score entries from BM25-based lists (A0, A2, A3) before Reciprocal Rank Fusion and leaves embedding lists whole. Standalone A0 and A3 results are unchanged, so every pinned number and the parity result stand. The filler is large on NFCorpus: 33.5% of A0's top-100 entries have score zero (0.1% on SciFact). Recall@100 is flagged soft for A0 and A3 on NFCorpus only; fused setups and A4 have no filler.
Why it matters: Rank-based fusion would otherwise give arbitrary documents real credit.
Evidence: tests/test_ranker.py (hand-computed RRF, determinism and list-order independence, tie order, zero-score exclusion), src/thedu/ranker/__init__.py
Next: RRF k and weights are untouched; they are M4's to tune.
Post: no

## 2026-10-27 · decision · M3 part 2: three rules, written from SciFact only and frozen before NFCorpus is looked at
Who: human + Claude
What: On SciFact dev (809 questions) the labels (rewrite-wins A3 beats A4 by 0.1 or more; embed-wins the reverse; both-fail; tie) come out as: embed-wins 188, rewrite-wins 124, tie 424, both-fail 73; A3 minus A4 overall -0.049 (range -0.072 to -0.027). Labels are stable: regenerating A3's rewrites for a fixed 100-question sample, twice, kept the same label for 94 and 95 of 100 questions against the cached ones, and 93 between the two fresh runs; 19 of the 20 cached rewrite-wins stayed rewrite-wins both times. Rewrite text changed more than labels did (identical for only 48 and 37 of 100). Two providers served the fresh calls, DeepInfra and NextBit. Between the two fresh runs the rewrite changed for 98% of questions (41 of 42) where the provider switched, and for 19% (11 of 58) where it did not, so most rewrite variation is provider switching, not sampling noise. Original cached rewrites have no provider on record. In the exploration, the rewrite's disadvantage against embeddings concentrates in questions where BM25 is unsure or the terms are common: by tercile, A3 minus A4 is -0.144 / -0.023 / +0.018 for BM25 top-1 gap (low / mid / high), -0.106 / -0.056 / +0.014 for mean IDF, and -0.145 / -0.008 / +0.022 for max IDF; A5 minus A4 follows the same order (-0.042 / +0.017 / +0.021 for the gap). Length, digits/codes/acronyms and vocabulary share show nothing (vocabulary share is 1.0 for every SciFact question, so it cannot be used). Post-hoc, the word overlap between the question and its best relevant document is lowest exactly where embeddings win (A3 minus A4 -0.114 in the lowest third), the reverse of "the rewrite fixes word gaps"; it uses the answer key and is not available to a router. Rules, frozen in configs/m3b_rules.yaml (`thedu freeze --add`, checksum in tests/frozen.sha256): R1 gap_ratio >= 0.286 (BM25 is confident); R2 mean_idf_norm >= 0.474 (terms are rare on average); R3 max_idf_norm >= 0.761 (at least one fairly rare term). Each predicts the rewrite does better against embeddings inside the group than outside. In-sample on SciFact (optimistic, the thresholds come from the same data): R1 group 270 questions (33%), in-vs-out A3 minus A4 +0.109 (+0.072 to +0.148); R2 270 (33%), +0.095 (+0.052 to +0.140); R3 549 (68%), +0.149 (+0.104 to +0.197). A rule holds on NFCorpus if the 95% range of that in-vs-out difference is above zero and the group is at least 15% of the questions. A5 minus A4 is reported as a secondary check. No other rules will be tried.
Why it matters: This is the pre-registration. The direction is the opposite of the naive guess (pay for the rewrite when BM25 is unsure), and NFCorpus is a different regime, where BM25 finds nothing for a third of the questions and the rewrite helps most. I do not know which way it will go, and a failure to hold is a real possible outcome. The three rules are not independent: R1 to R3 all measure how well BM25 can latch on to the question.
Evidence: runs/analysis/explore-scifact.json, runs/analysis/features-scifact.jsonl, runs/analysis/stability.json; stability runs ced207d5ec (llm seed 7) and 0cb87d9dee (llm seed 8), with their own cache in runs/analysis/stability/llm.jsonl, built from configs/m3b_stability_s7.yaml and s8; M3 run 2a4eb7b8d0.
Next: `thedu analyze confirm` on NFCorpus. A router is fitted on SciFact only if at least one rule holds; the main version chooses between A4 and A5, with A4 vs A3 alongside.
Post: maybe, as an honest "we wrote the rules down before the test" aside

## 2026-10-27 · finding · M3 part 2: one of three rules held on NFCorpus, and a one-feature router matches A5 with a third fewer rewrite calls
Who: human + Claude
What: Frozen rules tested unchanged on NFCorpus dev (324 questions; labels: tie 104, embed-wins 82, rewrite-wins 70, both-fail 68). Criterion: 95% range of (mean A3-A4 inside the group minus outside) above zero, group at least 15%. R1 (BM25 confident, gap_ratio >= 0.286) did NOT hold: group 35 questions (11%, below the 15% floor), +0.019 (-0.059 to +0.099). R2 (rare terms on average, mean_idf_norm >= 0.474) HELD: group 204 (63%), +0.057 (+0.016 to +0.099); secondary A5-A4 in vs out +0.033 (+0.003 to +0.063). R3 (max_idf_norm >= 0.761) did NOT hold: group 142 (44%), +0.035 (-0.008 to +0.078), same direction as SciFact but the range includes zero. Router, fit on SciFact pre-search features only and restricted to the feature of the rule that held (mean_idf_norm): fire the paid option when mean_idf_norm >= 0.448. Main version A4 vs A5 (pay for the rewrite, fused with embeddings, or not): on NFCorpus nDCG@10 0.354 (0.320 to 0.388) against A4 0.337, H0 0.336, A5 0.351; it calls the LLM on 68% of questions. Paired: vs A4 +0.0170 (+0.0016 to +0.0315), wins 99, losses 62; vs H0 +0.0175 (+0.0068 to +0.0281); vs A5 +0.0025 (-0.0043 to +0.0097), wins 35, losses 31, a tie. Rewrite cost per search (estimated) $0.000057 against A5's $0.000084, 32% less. On SciFact (5-fold cross-validated, never fit on the questions it scores) the same router gains +0.0164 nDCG@10 over always using A4, while always paying (A5) gains -0.0011; it fires on 45% of SciFact questions. Alongside, A4 vs A3 (choose A3 or A4 per question, fire at >= 0.469): NFCorpus 0.340, calls the LLM on 64%; vs A4 +0.0032 (-0.0156 to +0.0207), no gain; vs A3 +0.0179 (+0.0069 to +0.0305). Ceilings on NFCorpus: best of A4 or A5 per question 0.3896, best of A3 or A4 0.3941; the router recovers about a third of the 5.3-point gap from A4 to the first (A5 recovers about 28%).
Why it matters: The question was when to pay for the rewrite. On both datasets paying is worth it when the question's terms are, on average, rare in the collection, and not when they are common (SciFact: A5 minus A4 -0.036 / -0.012 / +0.044 by mean-IDF tercile). A router on that single feature keeps A5's NFCorpus quality and skips the rewrite on a third of questions, so it is cheaper and faster for the same accuracy. It does not beat A5 on accuracy.
Evidence: runs/analysis/confirm-nfcorpus.json, runs/analysis/route-nfcorpus.json, runs/analysis/explore-scifact.json; runs 2a4eb7b8d0 (SciFact) and f8a880adfa (NFCorpus). Stability: labels survive regeneration (same label for 93 to 95 of 100 questions; 18 to 19 of 19 to 20 rewrite-wins stay rewrite-wins), and rewrite text changes mostly when the serving provider changes (98% of questions with a provider switch between the two fresh runs, 19% without), details in the earlier decision entry and runs/analysis/stability.json.
Next: Cautions, in order of weight. (1) The router's feature was chosen after seeing that R2 held on NFCorpus, so NFCorpus is no longer a clean test of this router: the NFCorpus numbers are somewhat optimistic. One rule of three holding at a lower bound of +0.016 is also within what luck could produce (the rules are correlated, but three were tried). The honest next test is a dataset not used so far: FiQA once it has a verified embedding instruction, or the NFCorpus and SciFact test splits in M7. (2) The threshold was fit on SciFact and moved to NFCorpus unchanged, which worked here because the feature is scale-free; it may not move elsewhere. (3) The comparison with A5 is a tie, not a win, so the claim is "same quality, a third less rewrite cost", with a range of about one point either way. (4) Labels come from one generation of each rewrite; the stability run says they are robust at the label level, not at the level of individual rewrites. (5) Latency for skipped questions was not measured; the saving is proportional to the share skipped.
Post: yes, "pay for the rewrite only when the question has rare terms" is a practical result

## 2026-10-27 · decision · Pin the rewrite model to one OpenRouter provider: DeepInfra (fp8)
Who: human + Claude
What: New configs can set `llm.provider: {order: [DeepInfra], allow_fallbacks: false}`. It is sent as OpenRouter routing options, becomes part of the cache key only when set (so pinned calls never reuse unpinned answers), is used in the model check before a run, and the gateway raises if a response names a different provider. Why: in the label-stability run, rewrites changed for 98% of questions (41 of 42) when the serving provider switched and for 19% (11 of 58) when it did not, so provider mix, not sampling, was the main source of rewrite variation. OpenRouter lists two endpoints for qwen/qwen3-14b: NextBit (int4, $0.10 / $0.22 per million tokens) and DeepInfra (fp8, $0.12 / $0.24). Chosen: DeepInfra. Trade-off, on the record: the evidence for continuity with the cache points the other way. NextBit served 116 of the 200 stability calls and its fresh rewrites matched the original cached text 52% of the time against 30% for DeepInfra, so the spike's cached rewrites were probably served mostly by NextBit (the cached calls carry no provider record, so this is inference). I chose precision: the post should describe the model near full precision, and DeepInfra's price is the reference rate already in configs/pricing.yaml ($0.12 / $0.24), so cost needs no adjustment. The cost of that choice is that the cached SciFact and NFCorpus rewrites are not from the pinned provider, so A3 and A5 on those two datasets get re-run with it after FiQA, and every number in the post will come from one provider. A live call pinned to DeepInfra was served by DeepInfra.
Why it matters: A rewrite result should not depend on which backend happened to answer.
Evidence: src/thedu/llm.py, src/thedu/config.py, tests/test_llm.py; runs/analysis/stability.json; OpenRouter endpoints listing for qwen/qwen3-14b read 2026-10-27
Next: The default stays unpinned so existing configs and cached runs behave exactly as before. Champion and challenger are unpinned and read the spike cache; pin them when they are next regenerated (M4).
Post: yes, as a short "the same model is not the same model" note

## 2026-10-27 · decision · The router is pre-registered for a clean test on FiQA
Who: human + Claude
What: `configs/router.yaml`, frozen with `thedu freeze --add` (checksum in tests/frozen.sha256), fixes the router before any FiQA retrieval result has been seen: fire A5 (rewrite fused with embeddings) when mean_idf_norm >= 0.448, otherwise use A4 (embeddings only). The SciFact fit was 0.44847520450042516; 0.448 is frozen by decision, a negligible difference (on NFCorpus 222 of 324 questions fire at 0.448; the paired result against A4 is unchanged at +0.0170). Success criterion: the paired 95% range of (router minus A4) nDCG@10 on FiQA dev is above zero. Reported alongside, but not part of the verdict: router vs H0 and vs A5, the share of questions calling the LLM, rewrite cost per search. If the rule fires on no question the router equals A4 and the criterion fails by definition. Secondary, labelled as such: the three rules in configs/m3b_rules.yaml (already frozen, no new freedom) are also evaluated on FiQA. What has been looked at on FiQA so far: document length statistics (57,638 documents; median 117 tokens, 4.0% over the 512-token cap), the dev question count (500), and the MTEB task instruction text. No FiQA retrieval, label or feature result. Honest status of the router: its feature was chosen after R2 held on NFCorpus, and NFCorpus is therefore not a clean test of it; FiQA is.
Why it matters: A router that is tested on the data it was picked from proves little; this one gets one unseen test with the verdict rule fixed beforehand.
Evidence: configs/router.yaml, runs/analysis/route-nfcorpus.json (the fit), tests/test_analysis.py
Next: Embed FiQA, run A0, A3, A4, H0 and A5 on all 500 dev questions with the pinned provider, then `thedu analyze router --dataset fiqa`.
Post: maybe

## 2026-10-28 · finding · FiQA: the rewrite does not help, and the pre-registered router did not pass
Who: human + Claude
What: FiQA dev, 500 questions, fresh rewrites from DeepInfra (run 890ad993c9). nDCG@10: A0 0.2586, A3 0.2467, A4 0.4291, H0 0.3778, A5 0.3734. A4 beats A5 by 0.0557 (range 0.0318 to 0.0808, paired) and H0 by 0.0513; A5 vs H0 is -0.0044 (range -0.0156 to +0.0063). The frozen router (fire A5 when mean_idf_norm >= 0.448, else A4) called the LLM on 15 of 500 questions (3%) and scored +0.0017 against A4 (range -0.0015 to +0.0059): criterion NOT MET (lower bound must be above 0). Secondary rules R1 to R3 did not hold either; their groups are 16, 9 and 52 questions. Labels: embed-wins 244, rewrite-wins 51, tie 72, both-fail 133.
Why it matters: On a third corpus the rewrite adds nothing to embeddings (SciFact: no gain; NFCorpus: small gain; FiQA: clear loss), and the router's feature does not transfer: FiQA questions are plain words, so almost none pass the 0.448 threshold and the router falls back to A4. It is safe here (it matches A4) but it is not evidence that "rare terms" predicts when rewriting pays.
Evidence: runs/890ad993c9, runs/analysis/router-frozen-fiqa.json, runs/analysis/confirm-fiqa.json, configs/m3c_fiqa.yaml. FiQA A4 is not parity-checked against published scores (that needs the test answers, not read until M7); MTEB reports 0.46612 on the test split, for context only.
Next: Report as is. No threshold or feature changes after this result.
Post: yes, the pre-registered failure is part of the story

## 2026-10-28 · finding · Re-running A3 and A5 with the pinned provider changes no M3 conclusion
Who: human + Claude
What: Fresh DeepInfra rewrites instead of the cached ones (runs eed781f01f SciFact, e9b3631de6 NFCorpus). SciFact A3 0.6941 (was 0.6874), A5 0.7375 (was 0.7357); A5 vs H0 -0.0034 (range -0.0117 to +0.0049), still no win. NFCorpus A3 0.3205 (was 0.3221), A5 0.3506 (was 0.3514); A5 vs H0 +0.0142 (range +0.0028 to +0.0244), still a win (was +0.0150). Frozen router on NFCorpus: +0.0156 vs A4 (range +0.0006 to +0.0294), lower bound barely above zero, 69% LLM calls; this is not a clean test since the feature was picked after seeing NFCorpus.
Why it matters: Every post number can now come from one provider, and the choice of provider did not move any conclusion; differences are within 0.007.
Evidence: runs/eed781f01f, runs/e9b3631de6, runs/analysis/router-frozen-nfcorpus.json, configs/m3d_*_pinned.yaml
Next: Use the pinned runs for the post.
Post: yes, one line

## 2026-10-28 · decision · The goal is an embedding-free pipeline that matches or beats A4; "strict" rules
Who: human
What: (1) The project goal changes: build a pipeline without embeddings that matches or beats A4. Embeddings are the benchmark, not a component; H0 and A5 stay only as context. (2) "Strict" rules: a keyword index only, with no model used at index time (no document expansion). All intelligence is at query time: query rewriting and reranking are allowed. Document expansion (doc2query) goes to future work.
Why it matters: It makes the question sharp and the comparison fair: any gain has to come from what happens after the question arrives, and the index stays the same plain BM25 index everywhere.
Evidence: none yet; decided after the M3/M3c results (runs 890ad993c9, eed781f01f, e9b3631de6)
Next: Phase A diagnosis from existing runs (where the relevant documents sit, perfect-reranker ceiling), then the M5 design is re-cast around embedding-free arms (A0, A3, A0+reranker, A3+reranker) against A4.
Post: yes, this is the post's framing

## 2026-10-28 · finding · Phase A: a perfect reranker over BM25's top 100 would beat A4 on all three datasets, but FiQA's candidate pool is the limit
Who: agent
What: Dev, pinned-provider runs (SciFact eed781f01f 809 q, NFCorpus e9b3631de6 324 q, FiQA 890ad993c9 500 q). Share of questions whose best relevant document is in the top 10 / at ranks 11-100 / not in the top 100: SciFact A0 83.2/9.9/6.9%, A3 84.1/10.3/5.7%, A4 85.3/10.3/4.4%. NFCorpus A0 65.4/10.5/24.1%, A3 67.6/14.2/18.2%, A4 72.5/14.8/12.7%. FiQA A0 48.0/21.8/30.2%, A3 46.2/23.2/30.6%, A4 69.4/18.2/12.4%. nDCG@10 if the top 100 were ordered perfectly by the answer key (A0 / A3 / A4): SciFact 0.928 / 0.942 / 0.954 (actual A4 0.737); NFCorpus 0.522 / 0.626 / 0.661 (actual A4 0.337); FiQA 0.554 / 0.565 / 0.793 (actual A4 0.429). Perfect-rerank A0 minus actual A4: SciFact +0.191 (range 0.167 to 0.214), NFCorpus +0.185 (0.155 to 0.214), FiQA +0.125 (0.094 to 0.158); for A3: +0.205, +0.289, +0.136. To tie A4, a reranker over A0's candidates must capture about 18% (SciFact), 22% (NFCorpus) and 58% (FiQA) of the gap between A0's actual score and its ceiling (arithmetic from the figures above). The rewrite helps the candidate pool on NFCorpus (missing 24.1 to 18.2%) and a little on SciFact (6.9 to 5.7%), not on FiQA (30.2 to 30.6%). No relevant document sits in BM25's zero-score filler on NFCorpus, so excluding filler changes no ceiling.
Why it matters: On SciFact and NFCorpus the headroom is large and the share needed to catch A4 is small, so a reranker is a fair bet. On FiQA, 30% of questions have no relevant document in BM25's top 100, and a perfect reranker would still reach only 0.554, so a reranker must be very good (58% of the headroom) and the candidate pool is the real limit. These are ceilings, not predictions: no reranker is perfect, and the ceiling uses the answer key.
Evidence: runs/analysis/diagnosis-{scifact,nfcorpus,fiqa}.json; `thedu analyze diagnose`; src/thedu/analysis/diagnosis.py (new code path; earlier results untouched)
Next: Time the cross-encoder on keyword candidates (top 50 of A0 and A3), then re-cast M5. A candidate pool larger than 50 (top 100) is worth considering for FiQA.
Post: yes

## 2026-10-28 · decision · Reference price for reranking: Voyage rerank-2.5, $0.05 per million tokens
Who: agent
What: `configs/pricing_rerank.yaml` records the reference hosted rerank rate: Voyage rerank-2.5 at $0.05 per 1M tokens, billed on (query tokens x documents) + total document tokens. Source: https://docs.voyageai.com/docs/pricing, read 2026-10-28 (the page shows "Updated 29 days ago", no absolute date). The 200M free tokens are ignored. Cohere's own pricing page, read the same day, listed no per-search price for Rerank 3.5; about $2.00 per 1,000 searches comes from secondary sources only, kept as a cross-check.
Why it matters: A local reranker is never free in any reported number (AGENTS.md); this fixes the rate before any reranked result exists.
Evidence: configs/pricing_rerank.yaml
Next: Token counts will use the bge-reranker tokenizer, an approximation of Voyage's. Re-check the price before the post.
Post: no

## 2026-10-28 · decision · Pre-registration: cross-encoder reranking of keyword candidates (A0+R, A3+R) vs A4
Who: human + Claude
What: New setups A0R (A0+R) and A3R (A3+R), run on dev for NFCorpus, then SciFact, then FiQA, each saved as it finishes. Reranker: BAAI/bge-reranker-v2-m3, zero-shot, no tuning, max_length 512, query-time only. Candidates: the top 50 of the keyword list with score-zero filler dropped first (fewer than 50 if fewer match); the 50 are reordered by reranker score (ties keep the original order); ranks 51 to 100 stay below them in their original order, so Recall@100 is exactly the keyword list's. A3's rewrites come from the pinned-provider cache (runs/cache/llm.jsonl), so A3 is the A3 of runs eed781f01f / e9b3631de6 / 890ad993c9. **Primary test:** A3+R minus A4, paired nDCG@10, 95% range above zero, on each dataset separately (three tests, no correction). **Expectation from Phase A:** likely on SciFact and NFCorpus (a reranker has to capture about 18% and 22% of the headroom), unlikely on FiQA (30% of questions have no relevant document in BM25's top 100; ceiling 0.554 against A4's 0.429). **Secondary (labelled secondary in every report):** A0+R vs A4, A0+R vs A0, A3+R vs A3, A3+R vs A0. A4+R (does a reranker help embeddings too) is context only, later, lower priority. Cost per search = rewrite cost + rerank cost at the reference rate in configs/pricing_rerank.yaml ((query tokens x documents) + document tokens, bge tokenizer, each pair truncated at 512). Latency is recorded but this run shares the machine with other jobs, so its latency is inflated; the clean 20-question timings (5.7 to 5.9 s per question on SciFact, 3.8 to 5.8 s on NFCorpus) are the ones to quote.
Why it matters: Fixes the one test the post's answer rests on before any reranked number exists.
Evidence: runs/analysis/diagnosis-*.json (ceilings), configs/pricing_rerank.yaml; run folders to follow
Next: Implement RerankConfig and the two setups; run under caffeinate in the background.
Post: yes

## 2026-10-28 · decision · Pre-registration: keyword-index sweep, RM3, and embedding-filtered feedback (dev, three datasets, no LLM calls)
Who: human + Claude
What: **Keep rule for everything below:** a variant is kept only if its paired nDCG@10 range against current A0 (Snowball, k1 0.9, b 0.4, Lucene stopwords) is above zero (lower bound > 0) on at least two of three datasets. Recall@100 is reported with its paired range too (soft on NFCorpus for BM25-only lists), but the rule uses nDCG@10. **Index sweep, one change at a time:** stemmer Porter, none (Snowball is current); k1 in {0.6, 0.9, 1.2, 1.5} x b in {0.2, 0.4, 0.6, 0.75} (15 cells besides the current); title and body as separate fields, score = body score + w x title score, w in {0.5, 1, 2} (FiQA has no titles, so it cannot move there); word bigrams added to the index and the query; character n-grams of 3 to 5 inside words, alone and combined with words. If several variants pass, the one with the highest mean gain over the datasets is proposed; the grid is a tuning search on dev (16 cells), so its gain is optimistic, and A4 got no matching tuning (its settings came from parity), which the post must say. If one wins I propose, and do not run, re-running A3 and the Phase A ceilings on it. Synonym lists: future work. **RM3** (no model): feedback from A0's top 10 documents, document weights = their BM25 scores normalised to sum 1, term weights = sum over those documents of the document weight x (term frequency / document length) on stemmed, stopworded tokens; keep the top 15 terms, normalise to sum 1; final query weights = 0.5 x original query terms (each at 1/number of terms) + 0.5 x expansion terms. One setting, not tuned. Compared with A0 and A3 (paired): the baseline any LLM rewrite must beat. **Embedding-filtered feedback:** candidate terms = distinct lowercase word forms (3 or more letters, not stopwords, not already in the question) that occur in at least 2 of A0's top 20 documents; embed the terms (no instruction) and the question (with the dataset's task instruction, as A4 does) with Qwen3-Embedding-0.6B at query time; keep the 12 terms most similar to the question by cosine; search BM25 with the question x3 + those terms (the A3 construction). Compared with RM3, A3, A4 and A0 (paired). **This uses an embedding model at query time only, as a term filter: no document is embedded, no vectors are stored, the index is plain BM25.** Added latency per search is measured on 30 questions with no term-vector cache (the run itself caches term vectors across questions for speed). The embedding's reference price is not set, so its cost per search is reported as unknown.
Why it matters: These are the cheap baselines an LLM rewrite and a reranker have to beat, and the keep rule is fixed before any number exists.
Evidence: none yet
Next: Build `thedu sweep` (results with their config and code version in runs/sweeps/), then run.
Post: yes, as the "did we beat the obvious baselines" section

## 2026-10-28 · finding · Keyword-index sweep: no variant clears the pre-registered rule; the current A0 stays
Who: agent
What: 23 variants, one change at a time against current A0 (Snowball, k1 0.9, b 0.4), dev, paired nDCG@10 (SciFact 809, NFCorpus 324, FiQA 500 questions). Kept = range above zero on at least two datasets: none. No stemmer is worse everywhere (SciFact -0.031, NFCorpus -0.021, FiQA -0.019, all ranges below zero); Porter equals Snowball (within +-0.0005). Word bigrams hurt (-0.021, -0.006, -0.018). A separate title field (w 0.5, 1, 2) hurts SciFact (-0.000, -0.022, -0.071); NFCorpus +0.006, +0.004, -0.003 (ranges include zero); FiQA has no titles, so it cannot move. Character 3-5-grams: NFCorpus +0.027 (alone) and +0.029 (with words), Recall@100 +0.034 and +0.037, but SciFact -0.019 and -0.010, FiQA -0.031 and -0.023: one dataset of three. BM25 grid (15 cells): larger b helps FiQA only (k1 0.9: b 0.6 +0.008, b 0.75 +0.010; k1 1.2, b 0.6 +0.009; ranges above zero on FiQA alone), larger k1 helps NFCorpus a little (k1 1.5, b 0.4 +0.004; b 0.6 +0.004), neither beyond +0.010 and never on two datasets. The strongest single result, character n-grams on NFCorpus, sits on a Recall@100 that is soft for BM25-only lists (many questions match fewer than 100 documents), so read it with care.
Why it matters: The plain keyword index is not leaving an easy gain on the table; tuning it would not move the comparison with A4 (a hundredth of a point against gaps of 0.04 to 0.17). Nothing was adopted, so no matching tuning of A4 is owed and A3 and the Phase A ceilings are not re-run. Synonym lists remain future work.
Evidence: runs/sweeps/index-{scifact,nfcorpus,fiqa}.json (per-variant means, paired ranges for nDCG@10 and Recall@100, code version); `thedu sweep index`; src/thedu/retrievers/variants.py. The recomputed baseline matched the reference runs' A0 per question (checked in code).
Next: None for the index. 
Post: yes, one line ("we tried 23 index tweaks; none beat plain BM25 on two datasets")

## 2026-10-28 · finding · RM3 and embedding-filtered feedback: neither is kept; the LLM rewrite beats RM3 on all three datasets
Who: agent
What: Dev, paired nDCG@10, same questions. **RM3** (top 10 documents, 15 terms, original weight 0.5, one setting): vs A0 SciFact -0.018 (range -0.029 to -0.008), NFCorpus +0.018 (+0.009 to +0.029), FiQA -0.030 (-0.044 to -0.018): above zero on one of three, not kept. RM3 minus A3: -0.018, -0.018, -0.019 (every range below zero), so the LLM rewrite beats RM3 on all three datasets; on SciFact and FiQA that is because RM3 hurts, since A3 itself is level with A0 there. RM3 minus A4: -0.061, -0.034, -0.201. **Embedding-filtered feedback** (embedding model at query time only, as a term filter; no document vectors; 12 terms from the top 20 documents; question x3 + terms): vs A0 -0.013 (-0.025 to -0.002), +0.002 (-0.011 to +0.016), -0.059 (-0.077 to -0.041): not kept. vs RM3: +0.005 (ns), -0.016, -0.028; vs A3: -0.013 (-0.027 to +0.001), -0.034, -0.047; vs A4: -0.056, -0.051, -0.229. Its added latency over the first keyword search was 16,000x (SciFact), 29,000x (NFCorpus), 2,200x (FiQA) in a 30-question probe with no term-vector cache, on a machine shared with the reranker run, so these are inflated and relative only. The embedding's reference price is not set, so its cost is unknown.
Why it matters: The classic no-model expansion and the embedding-filtered variant do not reach the LLM rewrite, and certainly not A4; the filter picks topical neighbours ("mitochondrial", "organelle") rather than the words that find the answer. Note the baseline an LLM rewrite had to beat is beaten, but A3's own gain over A0 is small, so this is a low bar on SciFact and FiQA.
Evidence: runs/sweeps/rm3-*.json, runs/sweeps/efb-*.json (settings, per-question scores, example terms); `thedu sweep rm3`, `thedu sweep efb`; src/thedu/sweeps.py
Next: None queued. Latency of the filter should be re-timed on an idle machine if it goes in the post.
Post: yes

## 2026-10-28 · finding · Reranking keyword candidates: the primary test failed on all three datasets; A0+R matches A4 on SciFact
Who: agent
What: Pre-registered primary test (A3+R minus A4, paired nDCG@10, range above zero, per dataset) is NOT MET on any dataset. NFCorpus (324 q, run 8dbc179fa6): A3R 0.3240 vs A4 0.3369, -0.0129 (range -0.0335 to +0.0072). SciFact (809 q, run d8fe5f9aff): A3R 0.7387 vs A4 0.7368, +0.0019 (-0.0175 to +0.0213), a tie. FiQA (500 q, run 00fb868eaa): A3R 0.3772 vs A4 0.4291, -0.0519 (-0.0779 to -0.0255), A4 clearly ahead. The expectation "likely on SciFact and NFCorpus, unlikely on FiQA" was right only for FiQA; it was wrong for SciFact and NFCorpus. Secondary (labelled secondary): reranking helps a lot over its own candidates on SciFact (A0R vs A0 +0.0443, A3R vs A3 +0.0446) and FiQA (+0.1307, +0.1305), and modestly on NFCorpus (A0R +0.0292, A3R vs A3 +0.0035, no gain). A0R vs A4: SciFact +0.0019 (-0.0173 to +0.0217), NFCorpus -0.0235 (-0.0433 to -0.0036), FiQA -0.0398 (-0.0652 to -0.0147). What the rewrite adds on top of a reranker (A3R vs A0R): SciFact 0.0000, NFCorpus +0.0106 (+0.0017 to +0.0189), FiQA -0.0121 (-0.0254 to +0.0004). Cost per search: rewrite $0.00008 to $0.00009 plus rerank $0.0007 to $0.0010 (reference rate, bge tokenizer), so reranking costs about 8 to 12 times the rewrite. Fallbacks: none. Latency figures inside these runs are meaningless (machine shared with other jobs; relative values in the hundreds of thousands) and are not to be quoted; the clean timings are 3.8 to 5.9 s per question for 50 candidates.
Why it matters: Under the strict rules, an embedding-free pipeline matched A4 on SciFact and came close on NFCorpus, but did not beat it anywhere, and on FiQA it is 4 to 5 points behind, which is the candidate-pool limit Phase A predicted (30% of questions have no relevant document in BM25's top 100). On SciFact the cheapest arm (A0+R, no LLM) already equals A4, so the rewrite is not what gets there. The Phase A ceilings said a perfect reranker would clear A4 by 0.12 to 0.29; the zero-shot reranker captured about 19% of A0's headroom on SciFact (+0.0443 of 0.2330), 12% on NFCorpus (+0.0292 of 0.2377) and 44% on FiQA (+0.1307 of 0.2954); Phase A said 18%, 22% and 58% were needed to tie A4, so it got there on SciFact only.
Evidence: runs/8dbc179fa6 (NFCorpus), runs/d8fe5f9aff (SciFact), runs/00fb868eaa (FiQA); configs/rerank_*.yaml; pre-registration entry of 2026-10-28. The NFCorpus run recorded commit 49f1f08, the other two c6cf2cb; the changes between them are sweep code only, outside the pipeline.
Next: Decide with the human: deeper candidates (top 100) for FiQA, a stronger or tuned reranker, or A4+R as context (planned, lower priority). Nothing was tuned after seeing these results.
Post: yes

## 2026-10-28 · decision · M5 design v3: primary test named; FiQA yields 54 entities, not 200
Who: human + Claude
What: `docs/m5_codenames.md` now has the arms A0, A3, A0+R, A3+R and A0C (words plus character 3 to 5-grams) against A4, with H0/A5/router as context. **Primary test:** at 100% swapped, does any embedding-free arm match (lower bound of arm minus A4 above -0.01) or beat (above 0) A4; asked in condition A and in condition B, verdict reported for each; everything else is secondary (S1 A4 falls; S2 each arm's gap to A4 rises; lowest level at which an arm matches). Glossary: one short document per codename. Entities: FiQA dev questions name few things, so the list is 54 proper names (companies, agencies, exchanges, funds, countries and places, one person, named account types, currencies, websites) found by rule, filtered and described by the pinned model, then hand-corrected (merged duplicates, one wrong answer fixed, 20 descriptions rewritten so none contains a swapped name; every edit is in the note column); 26 generic terms (IPO, GDP, VAT, ...) are listed with keep=n. 92 of 500 dev questions contain at least one of the 54, and 14,647 documents change at 100%. With 92 questions the ranges are wide, so a failure to match is read as "A4 is better" only if the gap's upper bound is below 0.
Why it matters: The expectation and the primary test are fixed before any swapped-corpus number exists; the smaller entity count changes what the test can detect.
Evidence: docs/m5_codenames.md, configs/m5_entities.csv (draft, not frozen), src/thedu/m5/, src/thedu/prompts/m5_entities.txt
Next: Human reviews configs/m5_entities.csv; then freeze it, build the swap, run.
Post: yes

## 2026-10-28 · finding · Deeper candidates: relevant documents at ranks 51-100 are few, and the top-50 ceiling is already far above A4
Who: agent
What: From existing runs (no new retrieval), score-zero filler excluded. Relevant documents at ranks 51-100 (A0 / A3): SciFact 35 / 27 (4.1% / 3.0% of questions have any; 2.8% / 2.6% have them only there); NFCorpus 406 / 607 (33.6% / 51.9%; 2.5% / 4.0% only there, because NFCorpus questions have many relevant documents); FiQA 73 / 108 (13.2% / 18.0%; 4.0% / 7.2% only there). Perfect-rerank ceiling, top 50 vs top 100 (A0 / A3): SciFact 0.896 vs 0.928 / 0.915 vs 0.942; NFCorpus 0.482 vs 0.522 / 0.564 vs 0.626; FiQA 0.503 vs 0.554 / 0.487 vs 0.565. Actual A4: 0.737, 0.337, 0.429. The reranker run's A0+R / A3+R scores were 0.739 / 0.739 (SciFact), 0.313 / 0.324 (NFCorpus), 0.389 / 0.377 (FiQA).
Why it matters: Going to top 100 would lift the ceilings by 0.03 to 0.08, but the actual rerankers sit 0.16 to 0.22 below even the top-50 ceilings (SciFact 0.896 vs 0.739, NFCorpus 0.564 vs 0.324, FiQA 0.503 vs 0.389), so judge quality, not depth, is what limits them; doubling the candidates doubles rerank time and cost for a small ceiling gain. FiQA's A3 top-50 ceiling (0.487) is below A0's (0.503): the rewrite pushes some relevant documents down.
Evidence: runs/analysis/depth-{scifact,nfcorpus,fiqa}.json; `thedu analyze depth`
Next: A better judge (queued listwise LLM reranker on a fixed sample) before deeper candidates.
Post: no

## 2026-10-28 · decision · Queued, not run: LLM listwise reranker on a fixed sample, and A4+R overnight
Who: human + Claude
What: **(1) LLM listwise reranker**, to measure whether a better judge captures more headroom. Fixed 100-question sample per dataset (the same questions for every arm, drawn with a recorded seed). Candidates: top 50 of A0 and of A3, score-zero filler dropped, passages cut to about 100 words. One listwise prompt per question returning an order of the 50; the reranked top 50 sits above ranks 51-100 as in the bge run. Compared on the same 100 questions with A0+R / A3+R (bge), A4 and the Phase A top-50 ceiling; the measure is the share of headroom captured, (score - keyword baseline) / (top-50 ceiling - keyword baseline), with paired ranges. To be pre-registered (model, price, sample seed, primary test) before the first call. **Needs a decision first:** which hosted model and what budget. Rough size: 600 calls (100 questions x 3 datasets x 2 candidate sets) of about 8,000 input tokens is about 4.8M tokens, which is about $5 at $1 per million and about $14 at $3 per million, against the $10 total in configs/budget.yaml. **(2) A4+R**, context only: the bge reranker over A4's top 50 on all three datasets, configs/rerank_a4_{scifact,nfcorpus,fiqa}.yaml (written, not run); about 2.4 hours (SciFact 78 min, NFCorpus 27, FiQA 35 by the measured rates), overnight under caffeinate, with a completion notice this time.
Why it matters: Phase A and the depth check say judge quality, not candidate depth, is the limit on reranking; (1) tests that directly, (2) tells the post whether a reranker also lifts embeddings.
Evidence: runs/analysis/depth-*.json; configs/rerank_a4_*.yaml
Next: Human picks the model and budget for (1); say when to start (2).
Post: maybe

## 2026-10-28 · decision · M5 design v4 approved; entity list frozen
Who: human + Claude
What: `docs/m5_codenames.md` v4: A4+R added as an arm and benchmark; primary tests P1 (any embedding-free arm vs A4), P2 (A0+R vs A4+R), P3 (A3+R vs A4+R), all at 100% swapped, asked in condition A and in condition B, expectation supported only if it passes in both; "match" = lower bound of the gap above -0.01, "beat" = above 0; a miss counts as "the benchmark is better" only if the upper bound is below 0, otherwise inconclusive. Secondary S3, pre-registered: P1 to P3 repeated on questions whose entities are all organisations and products (types company, agency, exchange, fund, index, account, website, product, person, other: 37 entities) versus all places and currencies (country, region, place, city, currency: 17); mixed questions reported separately. `configs/m5_entities.csv` (54 keep=y, 26 keep=n generic terms kept as keep=n) approved as is and frozen with `thedu freeze --add` (checksum appended to tests/frozen.sha256). M5 runs after the A4+R runs finish.
Why it matters: The expectation, the benchmarks and the entity split are fixed before any swapped-corpus number exists.
Evidence: docs/m5_codenames.md, configs/m5_entities.csv, tests/frozen.sha256
Next: A4+R runs (next entry), then build the swap and run M5.
Post: yes

## 2026-10-28 · decision · Pre-registration: A4+R is the primary benchmark; A0+R and A3+R are tested against it
Who: human + Claude
What: A4+R (A4's top 50 reordered by bge-reranker-v2-m3, same settings as A0+R and A3+R, ranks 51-100 kept below) is run on SciFact, NFCorpus and FiQA dev (configs/rerank_a4_*.yaml, setups A4 and A4R). **Primary tests, per dataset (three each), paired nDCG@10 over all dev questions:** A0+R vs A4+R and A3+R vs A4+R, with A0+R and A3+R taken from runs 8dbc179fa6 (NFCorpus), d8fe5f9aff (SciFact), 00fb868eaa (FiQA). "Beat" = the 95% range of (keyword arm minus A4+R) lies above zero; "match" = its lower bound is above -0.01; the wide-range rule applies (a miss is "A4+R is better" only if the upper bound is below 0). **Expectation (mine, written before any A4+R number):** A4+R is at least level with A0+R and A3+R on every dataset because A4's candidate lists are better (Recall@50, computed from existing lists: SciFact A0 0.894, A3 0.914, A4 0.932; NFCorpus 0.200, 0.261, 0.271; FiQA 0.456, 0.446, 0.692); the keyword arms beat A4+R on no dataset and match it at most on SciFact (where A0+R already equalled A4 without a reranker) and possibly NFCorpus. Secondary: A4+R vs A4 (does a reranker lift embeddings, as it lifted BM25). The earlier primary test (A3+R vs A4, not met on any dataset) stands as reported. Run order NFCorpus, SciFact, FiQA, under caffeinate, with a completion notice; about 2.4 hours.
Why it matters: A reranker is a standard RAG stage, so the fair benchmark for a reranked keyword pipeline is a reranked embedding pipeline.
Evidence: configs/rerank_a4_*.yaml; runs/analysis/recall50-{scifact,nfcorpus,fiqa}.json (`thedu analyze recall50`)
Next: Report when the three runs finish; `thedu analyze paired` for the primary tests.
Post: yes

## 2026-10-28 · decision · Budget raised to $25; LLM listwise reranker proposed (no call made)
Who: human + Claude
What: configs/budget.yaml total_usd 10 to 25 (recorded spend so far $0.16). Proposal in docs/llm_rerank.md: `qwen/qwen3.5-397b-a17b` pinned to DeepInfra (fp8), $0.45 per 1M input and $3.00 per 1M output, from OpenRouter's endpoints list read 2026-10-28 (https://openrouter.ai/api/v1/models/qwen/qwen3.5-397b-a17b/endpoints); no Qwen model sits at $1 per million input, the neighbours being this one ($0.45) and `qwen/qwen3.7-max` ($1.475, Alibaba, precision unknown). Fixed 100-question sample per dataset (seed 7), top 50 of A0 and A3 reordered by one listwise prompt per question. Primary test per dataset: A3+L minus A3+R, paired range above zero. Expectation written before any call: a gain of at least 0.02 over bge on SciFact or NFCorpus, small on FiQA; within +-0.01 everywhere would mean the candidates, not the judge, are the limit. Estimated cost about $2.5 (reasoning off), up to about $9 if reasoning cannot be switched off.
Why it matters: Tests directly whether judge quality is the bottleneck that Phase A and the depth check point to.
Evidence: docs/llm_rerank.md; OpenRouter endpoints listing
Next: Human approves the model and the first (1-call) smoke test; then build and run.
Post: maybe

## 2026-10-28 · decision · LLM listwise reranker approved: qwen3.5-397b-a17b on DeepInfra (fp8); smoke test passed
Who: human + Claude
What: Approved by the human, with fallbacks off. Reasons: open weights, known precision (fp8), the same provider as the rewriter, and cheaper input (reranking is input-heavy; $0.45 per 1M input, $3.00 per 1M output, written to configs/pricing.yaml with the OpenRouter endpoints source read 2026-10-28). Additions from the human: each candidate cut to about 100 words so the prompt stays under about 10K tokens; if the ranking skips or repeats ids, the missing ones are appended in their original order and those fallbacks are counted per dataset in the report. Built: `llm.reasoning` option (sent to OpenRouter as `reasoning: {enabled: false}`, part of the cache key only when set), `ListwiseConfig`, setups A0L and A3L, `ListwiseReranker` (src/thedu/ranker/listwise.py, prompt listwise_v1), report fields `listwise_fallbacks`, `thedu analyze llm-rerank`. **Smoke test (one real call, SciFact sample question 1, 50 candidates):** served by DeepInfra, reasoning tokens 0 (reasoning switched off works), 8,408 input and 190 output tokens, $0.00435, 3.3 s, valid answer (no missing, repeated or invalid ids). Projected full cost: 600 calls x $0.00435 = about $2.6, not the worst case of about $9. The pre-registered tests in docs/llm_rerank.md are unchanged. The run waits for the A4+R runs to finish.
Why it matters: The cost question the human asked about (can reasoning be switched off on DeepInfra) is answered with a measurement: yes, and a call costs under half a cent.
Evidence: docs/llm_rerank.md; configs/llm_rerank_{scifact,nfcorpus,fiqa}.yaml; tests/test_listwise.py; runs/cache/llm.jsonl (the smoke call)
Next: Run the three sample configs after A4+R finishes; `thedu analyze llm-rerank` for the primary test.
Post: maybe

## 2026-10-28 · decision · Add A4L to the LLM reranker sample run; secondary tests A3L vs A4L and A0L vs A4L
Who: human + Claude
What: The 100-question sample runs also reorder A4's top 50 with the same LLM reranker (A4L), about 300 more calls, about $1.3. Pre-registered as secondary, alongside the unchanged primary A3L vs A3R: A3L minus A4L and A0L minus A4L, paired nDCG@10, with "match" (lower bound above -0.01) and "beat" (above 0) and the wide-range rule. The report adds cost per search and time per search for A0L, A3L, A4L next to A0R, A3R, A4R; embedding search is unpriced for every A4 arm. The cross-encoder runs shared the machine with other jobs, so a clean re-time of the cross-encoder on the same samples is run on an idle machine after the LLM runs.
Why it matters: A fair "same judge on both sides" comparison for the LLM, as A4+R is for the cross-encoder.
Evidence: docs/llm_rerank.md (addition), configs/llm_rerank_*.yaml (setups now include A4L)
Next: Run after A4+R finishes.
Post: maybe

## 2026-10-28 · decision · Tamil dataset added (IndicQARetrieval, ta), split by passage
Who: human (approval), agent (setup)
What: Added `indicqa_ta` from mteb/IndicQARetrieval at revision 51e8b328 (the one MTEB evaluates), CC0: 253 passages, 1,803 questions, 1,804 judgments, as expected. Split by passage, seed 7: 126 passages and 885 questions to dev, 127 passages and 917 questions to test; the corpus stays whole in both. One question has relevant passages on both sides and is in neither split. Frozen with `thedu freeze --add`: configs/indicqa_ta_split.json and the derived data/qrels/indicqa_ta/{dev,test}.tsv. Tamil keyword settings: PyStemmer Tamil, k1 0.9, b 0.4, no stopwords, an Indic-aware token pattern (the default one cuts Tamil words at every vowel sign). A4 instruction: "Retrieve text based on user query." (MTEB's retrieval-task default; IndicQARetrieval has no prompt of its own). No published Qwen3-Embedding-0.6B or BM25 score for Tamil IndicQARetrieval was found, so nothing was added to tests/expected.yaml and there is no external parity check.
Why it matters: A new language tests whether the rewrite-plus-rerank pipeline carries beyond English. Two things to know before the first run: at the 512-token limit A4 sees about a quarter of a median passage (every passage is over 512 Qwen tokens, median 2,029), and the Tamil stemmer both over- and under-strips (the empty-string stems are 0.28% of tokens).
Evidence: docs/tamil.md (pre-registration, written before any number), runs/analysis/indicqa_ta-stats.json. No run was made.
Next: your review of the stemmer output, stats and pre-registration; then wire the Tamil rewrite prompt and configs, and run dev.
Post: yes

## 2026-10-28 · learning · src/thedu/data/ was never in git
Who: agent
What: `.gitignore` had `data/`, which also matched the package `src/thedu/data/` (the loader and the guarded test-answers door), so those files were never committed. Changed the rule to `/data/` (the downloaded datasets folder only) and added the package.
Why it matters: A fresh clone could not import `thedu.data`; reproducibility needs the loader in the repo.
Evidence: `git ls-files src/thedu/data` was empty before this commit.
Next: none.
Post: no

## 2026-10-28 · decision · Tamil parked until the English runs and M5 are done
Who: human
What: The Tamil setup (loader, stats, frozen split, tests, draft docs/tamil.md) is committed as it is and parked. No Tamil run until the English runs (LLM listwise sample) and M5 are finished. docs/tamil.md is a draft, not a final pre-registration; the open decision is the max length for the embedding model and the reranker: 8,192 as primary or 512 (at 512 Qwen3 sees about a quarter of a median passage).
Why it matters: The pre-registration must not be treated as final while the length question is open.
Evidence: docs/tamil.md, runs/analysis/indicqa_ta-stats.json.
Next: English runs, then M5, then settle the length and finalise the Tamil pre-registration.
Post: no
