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
