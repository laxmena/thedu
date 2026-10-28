# AGENTS.md

Guide for coding agents working in this repo. Read it fully before your first change.

## What this repo is

Research code for one question: can an LLM that rewrites search queries over plain
keyword search (BM25) match embedding search, and at what cost?

Every number here ends up in a public post. Correct and reproducible beats fast and
feature-rich. This is not a library and has no users to support.

## Priorities, in order

When two of these conflict, the higher one wins.

1. **The judge is never wrong.** Never edit `src/thedu/judge/`, `data/qrels/`, or `tests/golden/`.
   If you believe they are wrong, stop and report. Do not "fix" them.
2. **No leakage.** Tune on dev splits only. Test answers load only through
   `thedu parity` (frozen baselines, once, in M1) and `thedu test-eval` (the final run).
   BRIGHT has no dev split: never tune on it.
3. **Reproducible.** Every result comes from a config file. Same config, same numbers.
4. **Fair.** Any tuning effort given to the planner is matched for the embedding baseline.
5. **Simple.** The smallest change that works. No new dependency without a stated reason.

## The system in one paragraph

A question goes in; ranked passages come out. The **planner** (an LLM) chooses which
words to search with and whether to search again. **Retrievers** (BM25, and embeddings
as the baseline) return candidates. The **ranker** merges and optionally reranks them.
The **judge** scores every run. Planner owns recall; ranker owns precision.

## The five parts and their contracts

| Part | Contract | Can change? |
| --- | --- | --- |
| Data | `load(name, split) -> Docs, Queries, Qrels` | No |
| Retrievers | `search(text: str, k: int) -> list[(doc_id, score)]`, sorted, len <= k | Settings only |
| Planner | `plan(question, hits | None) -> Plan` (JSON: `queries`, `must_keep`, `stop`) | Yes |
| Ranker | `rank(lists, question) -> list[(doc_id, score)]` | Yes |
| Judge | `score(run, qrels) -> Metrics` with 95% bootstrap ranges | No |

Do not add a sixth part without a human decision.

## What you may edit

| Path | Edit? |
| --- | --- |
| `src/thedu/planner/`, `src/thedu/prompts/`, `src/thedu/ranker/` | Yes |
| `configs/challenger.yaml` | Yes |
| `src/thedu/retrievers/` | Settings and bug fixes; keep the contract |
| `configs/champion.yaml` | Only through `thedu promote` |
| `src/thedu/judge/`, `data/qrels/`, `tests/golden/`, `tests/expected.yaml`, `tests/frozen.sha256` | Never |

Code lives in `src/thedu/` (the `thedu` package); part names elsewhere in this file
(`planner/`, `judge/`, ...) mean the subpackages there. Frozen files are read-only and
checksummed in `tests/frozen.sha256`; `thedu check` fails if one changes. `thedu freeze`
wrote that file once and refuses to run again. The one allowed change is
`thedu freeze --add PATH`, which appends a named file and never alters or removes a line;
log each addition in `JOURNAL.md`.

## How to make a change

Every change follows the same loop:

1. Change **one** thing: a prompt, a setting, or one function.
2. `thedu check`: fast tests, under a minute. Must pass.
3. `thedu dev configs/challenger.yaml`: runs dev queries and saves `runs/<hash>/`.
4. `thedu gate <hash>`: compares with the champion and prints PASS or FAIL.
5. On PASS, `thedu promote <hash>`. On FAIL, revert. Log the attempt either way.

**The gate passes in one of two ways:**

- **Better:** a paired bootstrap on the same dev questions shows the nDCG@10 gain over
  the champion is real (95%), and cost per search rises no more than 20%.
- **Cheaper:** quality holds (paired difference within ±0.5 nDCG@10) and cost per
  search falls.

Always compare paired, question by question. Two overlapping confidence ranges hide
real 1–2 point gains. Record how many attempts each promotion took: with many attempts,
some wins are luck. The unseen NFCorpus check catches those.

## Code rules

- Python 3.12, uv, type hints everywhere, pydantic for configs.
- Every command runs through one small CLI: `uv run thedu <command>` (Typer, in `cli.py`).
  Add a command there rather than a new script.
- Search is the default command: `thedu "question"` prints each round's rewrite, then
  the top results with sources. Flags: `--effort small|medium|large`, `--corpus`,
  `--compare` (embedding results side by side), `--json`. A known command name runs
  the command; anything else is a search. `thedu search <word>` forces a search.
- Scoring and ranking are pure functions. I/O stays at the edges.
- Every LLM call goes through `llm.call()`, which caches, counts tokens, and records
  cost. Never call a provider directly.
- Cost is always priced at a reference hosted rate for the model's size, even for
  local runs. Local runs are never "free" in any reported number.
- Report latency only as relative between setups, never as absolute laptop timings.
- Seeds, temperature, model names, and k values come from config. Never hard-code them.
- Secrets live only in `.env` (git-ignored). Never print, log, commit, or copy a key
  anywhere else, and never read `.env` except through the code that loads it.
- Prompts are plain text files in `src/thedu/prompts/`; the LLM returns search terms, not
  JSON (the spike's cached rewrites depend on it). The planner wraps them in a `Plan`
  (pydantic: `queries`, `must_keep`, `stop`), which must validate. On an empty or unusable
  answer: retry once, then fall back to the raw question and count the fallback. A run
  with more than 5% fallbacks refuses to report. JSON-output prompts come with the
  multi-round planner.
- A run records the git commit and a hash of the package source. Resuming a run under
  different code is refused (`--restart` sets the old run aside); a run never mixes two
  code versions.
- `must_keep` terms (IDs, codenames, quoted phrases) are copied exactly from the question.
- Notebooks only read `runs/`. They never produce numbers the post depends on.
- Log; don't print.

## Tests

| Command | What it runs | When |
| --- | --- | --- |
| `thedu check` | Unit, contract, schema, checksum, leakage tests, and the A0/A3 reproduction of the spike's SciFact numbers from cached rewrites | Every change |
| `thedu smoke` | A4 (embeddings) reproduction of the spike's SciFact number, and 20 questions end to end; under 2 minutes | Before `thedu dev` |
| `thedu parity` | Frozen baselines vs published scores, on test | Once in M1, and after any retriever change |
| `thedu all` | Every number in the post, from scratch | Before release |

A failing parity test blocks everything. Fix it before anything else.

## Stop and ask a human when

- Parity fails and two attempts haven't found why
- A change would touch a frozen path
- A result jumps more than 5 nDCG points: check for leakage first
- A run would exceed the budget in `configs/budget.yaml`
- You want to add a dependency, a part, or a dataset

## The journal

`JOURNAL.md` is the project's memory and the raw material for the post. Humans and
agents both write to it. Failed attempts and dead ends matter as much as wins:
they are what make the post honest.

**Write an entry when you:**

- Make a decision, or reverse one
- Try something that fails, and know why (or don't)
- Find something surprising in the results
- Get blocked
- Notice something worth remembering, even if it isn't acted on yet
- Promote a new champion

**Format:** newest at the bottom, one entry per thing.

```
## 2026-10-31 · finding · BM25Q beats plain BM25 on long questions
Who: agent | human
What: one or two sentences.
Why it matters: one sentence.
Evidence: runs/<hash>, or "none yet"
Next: what this changes, or what to try.
Post: yes | no   (would this make a good story in the post?)
```

Types: `decision`, `attempt`, `finding`, `learning`, `blocker`, `observation`.

**Rules:**

- Append only. Never edit or delete a past entry. If one was wrong, add a new entry
  that corrects it and links back.
- Plain words. Write it so a reader who wasn't there can follow it.
- Facts and evidence, not hype. "Recall@100 rose 4.1 points (range 2.3–5.9)", not
  "huge improvement".
- Keep a run's details in its run folder; the journal only links to it.

## After every run

The run folder (`runs/<hash>/`) is the record of what ran and what it scored. There is no
per-run notes file. If the run taught you something, or you made a decision because of it,
add a journal entry that links to the folder.
