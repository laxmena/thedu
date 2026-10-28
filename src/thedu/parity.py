"""`thedu parity`: frozen baselines vs published scores, on test. Built in M1.

One of only two modules allowed to import thedu.data.test_answers.
"""

from __future__ import annotations


def run_parity() -> None:
    from thedu.data.test_answers import load_test_answers  # noqa: F401 - the guarded door; used from M1

    raise NotImplementedError("thedu parity is built in M1")
