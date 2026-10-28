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
