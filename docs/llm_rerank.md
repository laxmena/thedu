# Proposal: LLM listwise reranker on a fixed sample

Status: APPROVED 2026-10-28 (model, provider, one smoke-test call). The smoke test ran; the 600-call run starts after the A4+R runs finish. Question: does a
better judge capture more of the headroom that the zero-shot bge cross-encoder left?

## Model and provider (prices read from OpenRouter on 2026-10-28)

- **Model:** `qwen/qwen3.5-397b-a17b` (open-weight, 397B total / 17B active, 262k context).
- **Provider, pinned:** DeepInfra, fp8, `allow_fallbacks: false`. $0.45 per 1M input, $3.00 per 1M output. Same provider as
  the rewrite model, supports `seed`. Source: https://openrouter.ai/api/v1/models/qwen/qwen3.5-397b-a17b/endpoints (cheapest
  endpoint is Alibaba at $0.39 / $2.34 with unknown precision; DeepInfra is the cheapest fp8 one).
- **Why not exactly $1:** nothing Qwen sits at $1 per million input. The nearest are this model ($0.45) and `qwen/qwen3.7-max`
  ($1.475 in, $4.425 out, Alibaba only, precision unknown). The 397B model is cheaper, open-weight and precision-labelled, so
  I propose it; the max model is the alternative if you want a flagship regardless of price.
- **Reasoning:** the model can think before answering, which could multiply the output tokens. The plan switches reasoning
  off through OpenRouter's `reasoning` option (a small gateway change, with a test). **Smoke test (2026-10-28):** one real SciFact question, 50 candidates: reasoning switched off (`reasoning_tokens` 0), served by DeepInfra, 8,408 input and 190 output tokens, $0.00435, 3.3 s, a valid ranking with no skipped or repeated ids. About $0.0044 per call times 600 calls is about $2.6.

## Design

- **Sample:** 100 dev questions per dataset, `select_qids(queries, 100, seed 7)`, the same questions for every arm.
- **Candidates:** the top 50 of A0 and of A3 (score-zero filler dropped), exactly as for bge. Passages cut to 100 words (a prompt is about 8,400 tokens, under the 10K ceiling).
- **Prompt:** the question and the 50 numbered passages in retrieval order; the model returns the passage numbers from most
  to least relevant, one line. Numbers it omits are appended in their original order; repeats and out-of-range numbers are ignored; each of these is counted and the counts are printed and saved per dataset (`listwise_fallbacks`: questions with a fallback, ids missing, repeated, invalid, failed calls). One call per
  question and candidate set, temperature 0, seed 7, max 1,000 output tokens. The reranked 50 sit above ranks 51-100.
- **Arms on the sample:** A0+L and A3+L (LLM), against A0+R and A3+R (bge, from runs 8dbc179fa6 / d8fe5f9aff / 00fb868eaa),
  A4, A4+R and the top-50 perfect-rerank ceiling, all restricted to the same 100 questions.
- **Measure:** nDCG@10, paired, 95% bootstrap; and **headroom captured** = (arm - keyword baseline) / (top-50 ceiling -
  keyword baseline), on the sample.

## Pre-registered tests

- **Primary, per dataset (three tests):** A3+L minus A3+R, paired nDCG@10, range above zero. The same for A0+L minus A0+R is
  secondary.
- **Secondary:** A3+L vs A4+R and A0+L vs A4+R ("match" lower bound above -0.01, "beat" above 0); headroom captured by L and by R.
- With 100 questions the ranges are wide (half-width about 0.04 to 0.07): a miss is "bge is as good or better" only if the
  upper bound of the difference is below 0; otherwise inconclusive.
- **Expectation (written before any call):** the larger model beats the cross-encoder on SciFact and NFCorpus (a gain of at
  least 0.02 on one of them); on FiQA the gain is small because the candidate pool, not the judge, limits it. If the gain is
  within +-0.01 everywhere, the judge is not the bottleneck and the candidates are.

## Size, cost and budget

- 600 calls (100 questions x 3 datasets x 2 candidate sets), each about 7,000 input tokens (50 passages of about 140 tokens plus
  the question and instructions): about 4.4M input tokens = **$2.0**.
- Output with reasoning off: about 250 tokens per call = 0.15M tokens = **$0.45**. Total about **$2.5**.
- Worst case, if reasoning cannot be switched off (about 4,000 output tokens per call): about $7 for output, about **$9** in total.
- Budget raised to **$25** in `configs/budget.yaml` (recorded spend so far $0.16); the gateway still refuses calls at the cap.
- Cost per search is reported at the DeepInfra price above; the reference rate is written to `configs/pricing.yaml` with this
  source and date before the run.
- Latency is not reported from this run; a listwise call over 50 passages takes seconds and is relative to A0 only.

## What gets built (after your OK)

A gateway option for `reasoning`, a listwise reranker behind the ranker contract (it only reorders its candidates), tests
(parsing, omitted and repeated numbers, reorder-only), configs for the three datasets, and a `thedu` command for the sample run.

## Addition (2026-10-28): A4L, same judge on both sides

The sample run also reorders A4's top 50 with the same LLM reranker (A4L), so "same judge on both sides" exists for the LLM
as it does for the cross-encoder (A4+R). **Secondary tests (pre-registered, labelled secondary):** A3+L minus A4+L and A0+L
minus A4+L, paired nDCG@10 on the 100-question samples, with the "match" (lower bound above -0.01) and "beat" (above 0) labels
and the wide-range rule. The primary test is unchanged (A3+L minus A3+R). About 300 more calls, about $1.3.
Reported next to the cross-encoder numbers: cost per search (rewrite + rerank at reference rates; embedding search is unpriced
for every A4 arm) and time per search. The cross-encoder runs shared the machine with other jobs, so their in-run times are
inflated; a clean re-time of the cross-encoder on the same samples is run on an idle machine after the LLM runs finish.
