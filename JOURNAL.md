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
