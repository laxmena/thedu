# Backlog (not run, not pre-registered yet)

- **300-word FiQA check for the LLM reranker.** The LLM judge saw each passage cut to about 100 words; the cross-encoder
  saw the whole passage (up to 512 tokens with the question). The FiQA and SciFact losses of A3L against A3R may come from the
  truncation, not the judge. Check: the same 100 FiQA sample questions, passages cut to about 300 words, same judge, A0L/A3L/A4L,
  pre-registered before the run. Added 2026-10-28.
