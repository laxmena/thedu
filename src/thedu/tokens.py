"""Offline token estimates for cached LLM calls that recorded no token split.

The prompt is counted as the chat-formatted user turn (template tokens included) and the answer as
its cached text plus the end-of-turn token. Hidden thinking tokens were stripped before caching, so
the estimate is a lower bound on what the provider billed. Qwen3-Embedding-0.6B ships the Qwen3 tokenizer.
"""

from __future__ import annotations

import logging
from functools import lru_cache

log = logging.getLogger(__name__)

TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"
_PROMPT_WRAP = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")  # 3 + 5 template tokens
_ANSWER_END = "<|im_end|>"  # 1 token


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


def estimate_tokens(prompt: str, answer: str) -> tuple[int, int]:
    """(input tokens, output tokens) for one chat call."""
    tok = _tokenizer()
    count = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])  # noqa: E731
    return count(_PROMPT_WRAP[0] + prompt + _PROMPT_WRAP[1]), count(answer + _ANSWER_END)
