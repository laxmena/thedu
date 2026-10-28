from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from thedu.llm import Usage


class Plan(BaseModel):
    """The planner's contract: what to search with, what to keep verbatim, whether to stop.

    `usage` and `fallback` are telemetry for the run registry, not part of the contract.
    """

    queries: list[str] = Field(min_length=1)
    must_keep: list[str] = []  # IDs, codenames, quoted phrases copied exactly from the question
    stop: bool = True
    usage: Usage = Field(default_factory=Usage, exclude=True)
    fallback: bool = Field(False, exclude=True)  # True: the LLM gave nothing usable; the raw question was used

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("queries")
    @classmethod
    def _non_blank(cls, v: list[str]) -> list[str]:
        if any(not q.strip() for q in v):
            raise ValueError("a query is blank")
        return v
