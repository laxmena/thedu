# Backlog (not run, not pre-registered yet)

- **300-word FiQA check for the LLM reranker.** The LLM judge saw each passage cut to about 100 words; the cross-encoder
  saw the whole passage (up to 512 tokens with the question). The FiQA and SciFact losses of A3L against A3R may come from the
  truncation, not the judge. Check: the same 100 FiQA sample questions, passages cut to about 300 words, same judge, A0L/A3L/A4L,
  pre-registered before the run. Added 2026-10-28.

- **Hypothesis: unknown words poison the query embedding (M5, S4 group of 22).** In the 22 affected questions whose relevant passage
  contains no entity, A0R and A3R beat unreranked A4 in condition A by about +0.20 although the passage never had the name, so
  the codename in the question seems to hurt A4 by itself. Possible test: for A4 only, drop the codename (or the whole entity phrase)
  from the question and see whether A4 recovers on those 22 and on the 72; same for A4+R. Pre-register before running. Added 2026-10-30.
