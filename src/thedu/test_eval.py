"""`thedu test-eval`: the final run on test. Built in M7.

One of only two modules allowed to import thedu.data.test_answers.
"""

from __future__ import annotations


def run_test_eval() -> None:
    from thedu.data.test_answers import load_test_answers  # noqa: F401 - the guarded door; used from M7

    raise NotImplementedError("thedu test-eval is built in M7")
